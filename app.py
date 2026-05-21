"""
FastAPI service for Armenian cursive OCR.
- /detect      : YOLOv8 word-level bounding boxes
- /recognize   : CRNN single-word transcription
- /ocr         : full pipeline (detect -> crop -> recognize -> stitch)
- /ocr/correct : OCR + LLM grammar correction (JSON response)
- /ocr/export  : OCR + LLM correction + smart file download (XLSX/CSV/PDF/TXT)
"""
import io
import logging
import os
import tempfile
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from ultralytics import YOLO

from llm_processor import correct_and_classify, OPENROUTER_API_KEY
from file_export import (
    export_table_xlsx,
    export_table_csv,
    export_text_pdf,
    export_text_txt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ config
DETECTOR_REPO = os.getenv("DETECTOR_REPO", "armvectores/yolov8n_handwritten_text_detection")
DETECTOR_FILE = os.getenv("DETECTOR_FILE", "best.pt")
CRNN_WEIGHTS  = os.getenv("CRNN_WEIGHTS", "crnn.pt")
API_TOKEN     = os.getenv("API_TOKEN")  # optional — leave unset for open access
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ CRNN
# EXACT architecture from the training notebook.
# Key differences from many online CRNN implementations:
#   - MaxPool layers 11,17 use kernel=(2,1) — halve height only, keep width
#   - Final conv (cnn.18) has NO padding — reduces height from 3→1
#   - forward() applies log_softmax

class BiLSTM(nn.Module):
    def __init__(self, in_sz: int, hid_sz: int, out_sz: int):
        super().__init__()
        self.lstm = nn.LSTM(in_sz, hid_sz, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hid_sz * 2, out_sz)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)


class CRNN(nn.Module):
    """
    Input : (B, 1, H, W)   where H = IMG_HEIGHT from checkpoint
    Output: (T, B, num_classes),  T ≈ W/4
    CNN height: H→H/2→H/4→H/8→H/16→1  (works for H in {48, 52, 56})
    """
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            # 48→24
            nn.Conv2d(1, 64, 3, padding=1),  nn.ReLU(True), nn.MaxPool2d(2, 2),
            # 24→12
            nn.Conv2d(64, 128, 3, padding=1),  nn.ReLU(True), nn.MaxPool2d(2, 2),
            # 12→12
            nn.Conv2d(128, 256, 3, padding=1),  nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  nn.ReLU(True),
            # 12→6
            nn.MaxPool2d((2, 1)),
            # 6→6
            nn.Conv2d(256, 512, 3, padding=1),  nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),  nn.ReLU(True),
            # 6→3
            nn.MaxPool2d((2, 1)),
            # 3→1
            nn.Conv2d(512, 512, kernel_size=3),  nn.ReLU(True),
        )
        self.rnn = nn.Sequential(
            BiLSTM(512, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"CNN height must be 1, got {h}. Check IMG_HEIGHT."
        return F.log_softmax(
            self.rnn(conv.squeeze(2).permute(2, 0, 1)), dim=2)


# ------------------------------------------------------------------ load
print("Loading detector…", flush=True)
detector_path = hf_hub_download(repo_id=DETECTOR_REPO, filename=DETECTOR_FILE,
                                cache_dir="/tmp/hf/hub")
detector = YOLO(detector_path)

print(f"Loading recognizer from {CRNN_WEIGHTS}…", flush=True)
ckpt = torch.load(CRNN_WEIGHTS, map_location="cpu", weights_only=False)
ALPHABET    = ckpt["alphabet"]                     # str or list of chars
HIDDEN_SIZE = int(ckpt.get("hidden_size", 256))
IMG_HEIGHT  = int(ckpt.get("img_height", 48))
NUM_CLASSES = int(ckpt.get("num_classes", len(ALPHABET) + 1))

# tokenizer: index 0 reserved for CTC blank, characters start at 1
CHARS = list(ALPHABET) if isinstance(ALPHABET, str) else list(ALPHABET)
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

recognizer = CRNN(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)
recognizer.load_state_dict(ckpt["model_state_dict"])
recognizer.eval()
print(f"Recognizer ready · alphabet={len(CHARS)} chars · hidden={HIDDEN_SIZE} · h={IMG_HEIGHT}", flush=True)


def ctc_greedy_decode(logits: torch.Tensor) -> str:
    """logits: [T, 1, C] (log_softmax output) -> string."""
    pred = logits.argmax(2).squeeze(1).cpu().tolist()
    out, prev = [], -1
    for p in pred:
        if p != prev and p != 0:
            out.append(IDX_TO_CHAR.get(p, ""))
        prev = p
    return "".join(out)


def preprocess_word(img: Image.Image) -> torch.Tensor:
    """Preprocess a word crop for CRNN recognition.

    Must match the training notebook's predict_word() exactly:
      1. Convert to RGB
      2. Resize the RGB image to (new_w, IMG_HEIGHT)
      3. Convert to grayscale (AFTER resize — this matters!)
      4. Normalize to [-1, 1]
    """
    img = img.convert("RGB")
    w, h = img.size
    new_w = max(4, int(w * (IMG_HEIGHT / h)))
    img = img.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    # Convert to grayscale AFTER resizing (matches notebook)
    g = img.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ------------------------------------------------------------------ detection helpers
# The YOLO detector was trained on grayscale document scans.
# Feeding it color photos causes false positives from grid lines, shadows, etc.

def _to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """Convert to grayscale then back to 3-channel (matches YOLO training data)."""
    gray = img.convert("L")
    return Image.merge("RGB", [gray, gray, gray])


def _detect_words(img: Image.Image, conf: float = 0.5):
    """Run YOLO detection on a grayscale version of the image via temp file.

    Returns (boxes_xyxy ndarray, confs ndarray).
    If no words found at the original orientation, tries ±90° rotation.
    """
    def _run(pil_img):
        gray_img = _to_grayscale_rgb(pil_img)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            gray_img.save(tmp_path, quality=95)
        try:
            res = detector.predict(source=tmp_path, conf=conf, verbose=False)
        finally:
            os.unlink(tmp_path)
        return res[0].boxes

    boxes = _run(img)

    # Auto-rotation: if nothing found, try rotating ±90°
    if len(boxes) == 0:
        best_count, best_angle = 0, 0
        for angle in [90, -90]:
            rotated = img.rotate(angle, expand=True)
            det = _run(rotated)
            if len(det) > best_count:
                best_count = len(det)
                best_angle = angle
        if best_count > 0:
            img = img.rotate(best_angle, expand=True)
            boxes = _run(img)

    xyxy = boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.empty((0, 4))
    confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else np.empty((0,))
    return img, xyxy, confs


# ------------------------------------------------------------------ API
app = FastAPI(
    title="Armenian Cursive OCR",
    description="Armenian cursive handwriting OCR with LLM post-processing and smart file export.",
    version="2.0.0",
)
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
    return {
        "status": "ok",
        "device": DEVICE,
        "alphabet_size": len(CHARS),
        "img_height": IMG_HEIGHT,
        "llm_enabled": bool(OPENROUTER_API_KEY),
    }


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)  # fix phone camera rotation
        return img
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    img, xyxy, confs = _detect_words(img)
    boxes: List[dict] = []
    for box, c in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": float(c)})
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
    img, xyxy, confs = _detect_words(img)
    W, H = img.size

    items = []
    for box, c in zip(xyxy, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        padding = 3
        x1c, y1c = max(0, x1 - padding), max(0, y1 - padding)
        x2c, y2c = min(W, x2 + padding), min(H, y2 + padding)
        crop = img.crop((x1c, y1c, x2c, y2c))
        with torch.no_grad():
            logits = recognizer(preprocess_word(crop))
        items.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": float(c),
                      "text": ctc_greedy_decode(logits)})

    # Group into lines: sort by reading order, then group by vertical proximity
    items.sort(key=lambda w: (round((w["y1"] + w["y2"]) / 2 / 25), w["x1"]))
    box_heights = [w["y2"] - w["y1"] for w in items]
    line_tol = float(np.median(box_heights) * 1.5) if box_heights else 20.0

    lines, current, current_y = [], [], None
    for w in items:
        cy = (w["y1"] + w["y2"]) / 2
        if current_y is not None and abs(cy - current_y) > line_tol:
            lines.append(" ".join(x["text"] for x in current))
            current, current_y = [], None
        current.append(w)
        current_y = cy if current_y is None else current_y
    if current:
        lines.append(" ".join(x["text"] for x in current))

    text = "\n".join(lines)
    return {"text": text, "words": items}


def _run_ocr_pipeline(img: Image.Image) -> dict:
    """Shared OCR pipeline used by /ocr, /ocr/correct, and /ocr/export."""
    img, xyxy, confs = _detect_words(img)
    W, H = img.size

    items = []
    for box, c in zip(xyxy, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        padding = 3
        x1c, y1c = max(0, x1 - padding), max(0, y1 - padding)
        x2c, y2c = min(W, x2 + padding), min(H, y2 + padding)
        crop = img.crop((x1c, y1c, x2c, y2c))
        with torch.no_grad():
            logits = recognizer(preprocess_word(crop))
        items.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": float(c),
                      "text": ctc_greedy_decode(logits)})

    # Group into lines
    items.sort(key=lambda w: (round((w["y1"] + w["y2"]) / 2 / 25), w["x1"]))
    box_heights = [w["y2"] - w["y1"] for w in items]
    line_tol = float(np.median(box_heights) * 1.5) if box_heights else 20.0

    lines, current, current_y = [], [], None
    for w in items:
        cy = (w["y1"] + w["y2"]) / 2
        if current_y is not None and abs(cy - current_y) > line_tol:
            lines.append(" ".join(x["text"] for x in current))
            current, current_y = [], None
        current.append(w)
        current_y = cy if current_y is None else current_y
    if current:
        lines.append(" ".join(x["text"] for x in current))

    text = "\n".join(lines)
    return {"text": text, "words": items}


# ------------------------------------------------------------------ LLM endpoints

@app.post("/ocr/correct")
async def ocr_correct(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Run OCR then send the text to an LLM for grammar correction.

    Returns JSON with corrected text and content-type classification.
    """
    _check_auth(authorization)
    img = _read_image(await file.read())
    ocr_result = _run_ocr_pipeline(img)
    raw_text = ocr_result["text"]

    llm_result = await correct_and_classify(raw_text)

    return {
        "raw_text": raw_text,
        "corrected_text": llm_result.get("corrected_text", raw_text),
        "content_type": llm_result.get("content_type", "text"),
        "rows": llm_result.get("rows"),
        "words": ocr_result["words"],
    }


# MIME types for each export format
_EXPORT_MIME = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv":  "text/csv; charset=utf-8",
    "pdf":  "application/pdf",
    "txt":  "text/plain; charset=utf-8",
}


@app.post("/ocr/export")
async def ocr_export(
    file: UploadFile = File(...),
    format: Optional[str] = Query(
        None,
        description=(
            "Desired output format: 'xlsx', 'csv', 'pdf', or 'txt'. "
            "If omitted the server auto-selects based on content type "
            "(xlsx for tables, pdf for text)."
        ),
    ),
    fix_grammar: bool = Query(
        True, description="Send OCR text to LLM for grammar correction."
    ),
    authorization: Optional[str] = Header(None),
):
    """Full pipeline: OCR → LLM correction → downloadable file.

    - Tables  → XLSX (or CSV if format='csv')
    - Text    → PDF  (or TXT if format='txt')
    """
    _check_auth(authorization)
    img = _read_image(await file.read())
    ocr_result = _run_ocr_pipeline(img)
    raw_text = ocr_result["text"]

    if not raw_text.strip():
        raise HTTPException(400, "No text detected in the image.")

    # --- LLM post-processing (optional) ---
    if fix_grammar:
        llm_result = await correct_and_classify(raw_text)
    else:
        llm_result = {"content_type": "text", "corrected_text": raw_text}

    content_type = llm_result.get("content_type", "text")
    corrected = llm_result.get("corrected_text", raw_text)
    rows = llm_result.get("rows")

    # --- Determine output format ---
    fmt = (format or "").lower().strip()
    if not fmt:
        fmt = "xlsx" if content_type == "table" and rows else "pdf"

    if fmt not in _EXPORT_MIME:
        raise HTTPException(
            400,
            f"Unsupported format '{fmt}'. Use one of: {', '.join(_EXPORT_MIME)}.",
        )

    # --- Generate file ---
    if fmt == "xlsx":
        if not rows:
            # If user forces xlsx but LLM didn't return rows, make a single-column table
            rows = [["Text"]] + [[line] for line in corrected.split("\n") if line.strip()]
        data = export_table_xlsx(rows)
        filename = "ocr_result.xlsx"

    elif fmt == "csv":
        if not rows:
            rows = [["Text"]] + [[line] for line in corrected.split("\n") if line.strip()]
        data = export_table_csv(rows)
        filename = "ocr_result.csv"

    elif fmt == "pdf":
        data = export_text_pdf(corrected)
        filename = "ocr_result.pdf"

    else:  # txt
        data = export_text_txt(corrected)
        filename = "ocr_result.txt"

    return Response(
        content=data,
        media_type=_EXPORT_MIME[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
