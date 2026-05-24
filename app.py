"""
FastAPI service for Armenian cursive OCR.
Uses the armenian-ocr module (CRAFT detection + ResNet-BiLSTM-Attn recognition).

Endpoints:
- /detect      : CRAFT word-level bounding boxes
- /recognize   : Single-word recognition
- /ocr         : Full pipeline (detect -> crop -> recognize -> stitch)
- /ocr/correct : OCR + LLM grammar correction (JSON response)
- /ocr/export  : OCR + LLM correction + smart file download (XLSX/CSV/PDF/TXT)
"""
import io
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Monkey-patch: newer torchvision removed `model_urls` from submodules.
# The armenian-ocr basenet imports it but never uses it (commented-out code).
# We inject an empty dict so the import doesn't crash.
# ---------------------------------------------------------------------------
import torchvision.models.vgg as _vgg_module

if not hasattr(_vgg_module, "model_urls"):
    _vgg_module.model_urls = {}

# ---------------------------------------------------------------------------
# Add armenian-ocr to the Python path so we can import `armenian_ocr`
# ---------------------------------------------------------------------------
_OCR_MODULE_DIR = os.path.join(os.path.dirname(__file__), "armenian-ocr")
if _OCR_MODULE_DIR not in sys.path:
    sys.path.insert(0, _OCR_MODULE_DIR)

from armenian_ocr import OcrWrapper  # noqa: E402

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from llm_processor import correct_and_classify, OPENROUTER_API_KEY  # noqa: E402
from file_export import (  # noqa: E402
    export_table_xlsx,
    export_table_csv,
    export_text_pdf,
    export_text_txt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ config
DET_MODEL_DIR = os.getenv("DET_MODEL_DIR", os.path.join(_OCR_MODULE_DIR, "detection"))
REC_MODEL_DIR = os.getenv("REC_MODEL_DIR", os.path.join(_OCR_MODULE_DIR, "recognition"))
API_TOKEN     = os.getenv("API_TOKEN")  # optional — leave unset for open access
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ load OCR
print("Loading armenian-ocr pipeline…", flush=True)
ocr = OcrWrapper()
ocr.load(det_model_dir=DET_MODEL_DIR, rec_model_dir=REC_MODEL_DIR, device=DEVICE)
ocr.rec_wrapper.opt.workers = 0  # Force 0 workers to prevent multiprocessing crash
print(f"OCR pipeline ready on {DEVICE}", flush=True)

# ------------------------------------------------------------------ API
app = FastAPI(
    title="Armenian Cursive OCR",
    description="Armenian cursive handwriting OCR with LLM post-processing and smart file export.",
    version="3.0.0",
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
        "detector": "CRAFT",
        "recognizer": "ResNet-BiLSTM-Attn",
        "llm_enabled": bool(OPENROUTER_API_KEY),
    }


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)  # fix phone camera rotation
        return img
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")


def _pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to numpy array (RGB, uint8)."""
    return np.array(img)


def _sort_words_into_lines(items: List[dict]) -> str:
    """Group word items into lines by vertical proximity, then join."""
    if not items:
        return ""

    # Sort by reading order: vertical center, then horizontal position
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

    return "\n".join(lines)


# ------------------------------------------------------------------ endpoints

@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    np_img = _pil_to_np(img)
    boxes = ocr.det_wrapper.predict(image=np_img)

    result_boxes: List[dict] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        result_boxes.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "conf": 1.0,
        })
    return {"width": img.width, "height": img.height, "boxes": result_boxes}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...),
                    authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    np_img = _pil_to_np(img)
    # Convert to grayscale for the recognition model
    from armenian_ocr import utils
    gray = utils.grayscale(np_img)
    texts = ocr.rec_wrapper.predict([gray])
    return {"text": texts[0] if texts else ""}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...),
                       authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    img = _read_image(await file.read())
    result = _run_ocr_pipeline(img)
    return result


def _run_ocr_pipeline(img: Image.Image) -> dict:
    """Shared OCR pipeline used by /ocr, /ocr/correct, and /ocr/export."""
    np_img = _pil_to_np(img)
    predictions = ocr.predict(image=np_img)

    items = []
    for pred in predictions:
        box = pred["box"]
        x1, y1, x2, y2 = box
        items.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "conf": 1.0,
            "text": pred["text"],
        })

    text = _sort_words_into_lines(items)
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
