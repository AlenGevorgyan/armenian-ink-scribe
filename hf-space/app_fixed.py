"""
FastAPI service for Armenian cursive OCR.
- /detect    : YOLOv8 word-level bounding boxes
- /recognize : CRNN single-word transcription
- /ocr       : full pipeline (detect -> crop -> recognize -> stitch)
"""
import io
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------ config
DETECTOR_REPO = os.getenv("DETECTOR_REPO", "armvectores/yolov8n_handwritten_text_detection")
DETECTOR_FILE = os.getenv("DETECTOR_FILE", "best.pt")
CRNN_WEIGHTS  = os.getenv("CRNN_WEIGHTS", "crnn.pt")
API_TOKEN     = os.getenv("API_TOKEN")  # optional — leave unset for open access
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ CRNN
class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.lstm = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.linear = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.lstm(x)
        T, b, h = recurrent.size()
        return self.linear(recurrent.view(T * b, h)).view(T, b, -1)


class CRNN(nn.Module):
    """Canonical CRNN (Shi et al.) — matches keys cnn.0/3/6/7/9/12/13/15/18 and rnn.0/1."""
    def __init__(self, num_classes: int, hidden_size: int = 256, in_channels: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 0
            nn.ReLU(True),                        # 1
            nn.MaxPool2d(2, 2),                   # 2
            nn.Conv2d(64, 128, 3, 1, 1),          # 3
            nn.ReLU(True),                        # 4
            nn.MaxPool2d(2, 2),                   # 5
            nn.Conv2d(128, 256, 3, 1, 1),         # 6
            nn.BatchNorm2d(256),                  # 7
            nn.ReLU(True),                        # 8
            nn.Conv2d(256, 256, 3, 1, 1),         # 9
            nn.ReLU(True),                        # 10
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 11
            nn.Conv2d(256, 512, 3, 1, 1),         # 12
            nn.BatchNorm2d(512),                  # 13
            nn.ReLU(True),                        # 14
            nn.Conv2d(512, 512, 3, 1, 1),         # 15
            nn.ReLU(True),                        # 16
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 17
            nn.Conv2d(512, 512, 2, 1, 0),         # 18
            nn.ReLU(True),                        # 19
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"expected feature height 1, got {h}"
        conv = conv.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        return self.rnn(conv)


# ------------------------------------------------------------------ load
print("Loading detector…", flush=True)
detector_path = hf_hub_download(repo_id=DETECTOR_REPO, filename=DETECTOR_FILE,
                                cache_dir="/tmp/hf/hub")
detector = YOLO(detector_path)

print(f"Loading recognizer from {CRNN_WEIGHTS}…", flush=True)
ckpt = torch.load(CRNN_WEIGHTS, map_location="cpu", weights_only=False)
ALPHABET    = ckpt["alphabet"]                     # str or list of chars
HIDDEN_SIZE = int(ckpt.get("hidden_size", 256))
IMG_HEIGHT  = int(ckpt.get("img_height", 32))
NUM_CLASSES = int(ckpt.get("num_classes", len(ALPHABET) + 1))

# tokenizer: index 0 reserved for CTC blank, characters start at 1
if isinstance(ALPHABET, str):
    CHARS = list(ALPHABET)
else:
    CHARS = list(ALPHABET)
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

recognizer = CRNN(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)
recognizer.load_state_dict(ckpt["model_state_dict"])
recognizer.eval()
print(f"Recognizer ready · alphabet={len(CHARS)} chars · hidden={HIDDEN_SIZE} · h={IMG_HEIGHT}", flush=True)


def ctc_greedy_decode(logits: torch.Tensor) -> str:
    """logits: [T, 1, C] -> string."""
    pred = logits.softmax(2).argmax(2).squeeze(1).cpu().tolist()
    out, prev = [], -1
    for p in pred:
        if p != prev and p != 0:
            out.append(IDX_TO_CHAR.get(p, ""))
        prev = p
    return "".join(out)


def preprocess_word(img: Image.Image) -> torch.Tensor:
    """Grayscale, resize to fixed height, pad/normalize, return [1,1,H,W]."""
    g = img.convert("L")
    w, h = g.size
    new_w = max(8, int(w * (IMG_HEIGHT / h)))
    g = g.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ------------------------------------------------------------------ API
app = FastAPI(title="Armenian Cursive OCR")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _check_auth(authorization: Optional[str]):
    if not API_TOKEN:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_TOKEN:
        raise HTTPException(401, "Invalid token")


@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE, "alphabet_size": len(CHARS),
            "img_height": IMG_HEIGHT}


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    res = detector.predict(np.array(img), verbose=False)[0]
    boxes: List[dict] = []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": float(b.conf[0])})
    return {"width": img.width, "height": img.height, "boxes": boxes}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...),
                    authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    with torch.no_grad():
        logits = recognizer(preprocess_word(img))
    return {"text": ctc_greedy_decode(logits)}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...),
              authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    res = detector.predict(np.array(img), verbose=False)[0]

    items = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        with torch.no_grad():
            logits = recognizer(preprocess_word(crop))
        items.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "text": ctc_greedy_decode(logits)})

    # Group into lines top-to-bottom, sort left-to-right within each line.
    items.sort(key=lambda w: w["y1"])
    lines, current, current_y = [], [], None
    LINE_TOL = IMG_HEIGHT  # rough line-height threshold in original pixels
    for w in items:
        if current_y is None or abs(w["y1"] - current_y) < LINE_TOL:
            current.append(w)
            current_y = w["y1"] if current_y is None else current_y
        else:
            lines.append(sorted(current, key=lambda x: x["x1"]))
            current, current_y = [w], w["y1"]
    if current:
        lines.append(sorted(current, key=lambda x: x["x1"]))

    text = "\n".join(" ".join(w["text"] for w in line) for line in lines)
    return {"text": text, "words": items}
