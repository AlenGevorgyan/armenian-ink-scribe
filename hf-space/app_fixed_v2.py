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
from PIL import Image, ImageOps
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
    """Matches training notebook exactly. Height flow at h=48: 48→24→12→6→3→1."""
    def __init__(self, num_classes: int, hidden_size: int = 256, in_channels: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),   # 48→24
            nn.Conv2d(64, 128, 3, padding=1),         nn.ReLU(True), nn.MaxPool2d(2, 2),   # 24→12
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),                       nn.ReLU(True),
            nn.MaxPool2d((2, 1)),                                                          # 12→6
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),                       nn.ReLU(True),
            nn.MaxPool2d((2, 1)),                                                          # 6→3
            nn.Conv2d(512, 512, kernel_size=3),                      nn.ReLU(True),        # 3→1 (no padding)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"CNN height must be 1, got {h}"
        conv = conv.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        return self.rnn(conv)


# ------------------------------------------------------------------ load
print("Loading detector…", flush=True)
detector_path = hf_hub_download(repo_id=DETECTOR_REPO, filename=DETECTOR_FILE,
                                cache_dir="/tmp/hf/hub")
detector = YOLO(detector_path)

print(f"Loading recognizer from {CRNN_WEIGHTS}…", flush=True)
ckpt = torch.load(CRNN_WEIGHTS, map_location="cpu", weights_only=False)
print(f"[ckpt] keys = {list(ckpt.keys())}", flush=True)

ALPHABET    = ckpt.get("alphabet")
CHAR_TO_IDX = ckpt.get("char_to_idx") or ckpt.get("char2idx") or ckpt.get("vocab")
HIDDEN_SIZE = int(ckpt.get("hidden_size", 256))
IMG_HEIGHT  = int(ckpt.get("img_height", 32))

# Build IDX_TO_CHAR from whichever mapping is present.
if CHAR_TO_IDX:
    IDX_TO_CHAR = {int(i): c for c, i in CHAR_TO_IDX.items()}
    NUM_CLASSES = int(ckpt.get("num_classes", max(IDX_TO_CHAR) + 1))
    BLANK_IDX = 0 if 0 not in IDX_TO_CHAR else (NUM_CLASSES - 1)
else:
    if isinstance(ALPHABET, str):
        CHARS = list(ALPHABET)
    else:
        CHARS = list(ALPHABET or [])
    # Default convention: blank=0, chars at 1..N
    IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
    NUM_CLASSES = int(ckpt.get("num_classes", len(CHARS) + 1))
    BLANK_IDX = 0

print(f"[ckpt] num_classes={NUM_CLASSES} blank_idx={BLANK_IDX} "
      f"alphabet_len={len(IDX_TO_CHAR)} sample={list(IDX_TO_CHAR.items())[:5]}",
      flush=True)

recognizer = CRNN(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)
recognizer.load_state_dict(ckpt["model_state_dict"])
recognizer.eval()
print(f"Recognizer ready · alphabet={len(CHARS)} chars · hidden={HIDDEN_SIZE} · h={IMG_HEIGHT}", flush=True)


def ctc_greedy_decode(logits: torch.Tensor) -> str:
    """logits: [T, 1, C] -> string."""
    pred = logits.softmax(2).argmax(2).squeeze(1).cpu().tolist()
    out, prev = [], -1
    for p in pred:
        if p != prev and p != BLANK_IDX:
            ch = IDX_TO_CHAR.get(p)
            if ch is not None:
                out.append(ch)
        prev = p
    return "".join(out)


def ctc_greedy_decode(logits: torch.Tensor, debug: bool = False) -> str:
    """logits: [T, 1, C] -> string."""
    pred = logits.softmax(2).argmax(2).squeeze(1).cpu().tolist()
    if debug:
        from collections import Counter
        print(f"[decode] T={len(pred)} unique={Counter(pred).most_common(8)}", flush=True)
    out, prev = [], -1
    for p in pred:
        if p != prev and p != BLANK_IDX:
            ch = IDX_TO_CHAR.get(p)
            if ch is not None:
                out.append(ch)
        prev = p
    return "".join(out)


def preprocess_word(img: Image.Image, invert: bool = False, norm: str = "tanh") -> torch.Tensor:
    """Grayscale, resize to fixed height, normalize, return [1,1,H,W].

    norm:
      "tanh"     -> (x - 0.5) / 0.5  (range [-1, 1])
      "unit"     -> x / 255          (range [0, 1])
      "imagenet" -> (x - 0.485) / 0.229  (single-channel ImageNet)
    """
    g = img.convert("L")
    w, h = g.size
    new_w = max(8, int(w * (IMG_HEIGHT / h)))
    g = g.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    if norm == "tanh":
        arr = (arr - 0.5) / 0.5
    elif norm == "imagenet":
        arr = (arr - 0.485) / 0.229
    # "unit" -> leave as-is
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
    return {"status": "ok", "device": DEVICE,
            "alphabet_size": len(IDX_TO_CHAR),
            "num_classes": NUM_CLASSES, "blank_idx": BLANK_IDX,
            "img_height": IMG_HEIGHT}


@app.get("/debug")
def debug():
    return {
        "num_classes": NUM_CLASSES,
        "blank_idx": BLANK_IDX,
        "img_height": IMG_HEIGHT,
        "hidden_size": HIDDEN_SIZE,
        "alphabet_size": len(IDX_TO_CHAR),
        "alphabet_sample": list(IDX_TO_CHAR.items())[:20],
    }


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
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
        logits = recognizer(preprocess_word(img, invert=False, norm="tanh"))
    text = ctc_greedy_decode(logits, debug=True)
    print(f"[recognize] text={text!r}", flush=True)
    return {"text": text}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...),
              authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    res = detector.predict(np.array(img), verbose=False, conf=0.10, iou=0.45)[0]

    items = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
        conf = float(b.conf[0])
        pad_x = max(2, int((x2 - x1) * 0.08))
        pad_y = max(2, int((y2 - y1) * 0.18))
        crop_box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img.width, x2 + pad_x),
            min(img.height, y2 + pad_y),
        )
        crop = img.crop(crop_box)
        with torch.no_grad():
            logits = recognizer(preprocess_word(crop))
        txt = ctc_greedy_decode(logits)
        items.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "h": y2 - y1, "conf": conf, "text": txt})

    print(
        f"[ocr] image={img.width}x{img.height} detected={len(items)} "
        f"non_empty={sum(1 for w in items if w['text'].strip())} "
        f"sample={[(w['conf'], w['text']) for w in items[:10]]}",
        flush=True,
    )

    # Drop boxes where the recognizer produced nothing — they only add noise.
    items = [w for w in items if w["text"].strip()]

    if not items:
        return {"text": "", "words": [], "detected": len(res.boxes)}

    # Group into lines top-to-bottom using a tolerance derived from actual box heights.
    items.sort(key=lambda w: w["y1"])
    median_h = sorted(w["h"] for w in items)[len(items) // 2]
    LINE_TOL = max(8, int(median_h * 0.6))

    lines, current, current_y = [], [], None
    for w in items:
        if current_y is None or abs(w["y1"] - current_y) < LINE_TOL:
            current.append(w)
            if current_y is None:
                current_y = w["y1"]
        else:
            lines.append(sorted(current, key=lambda x: x["x1"]))
            current, current_y = [w], w["y1"]
    if current:
        lines.append(sorted(current, key=lambda x: x["x1"]))

    text = "\n".join(" ".join(w["text"] for w in line) for line in lines)
    return {"text": text, "words": items}
