"""
OpenRouter LLM integration for Armenian OCR post-processing.

Sends raw OCR text to an LLM to:
  1. Fix grammatical / spelling errors in Armenian text
  2. Classify whether the content is a "table" or "text"
  3. If table → return structured rows as JSON
  4. If text  → return corrected prose
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# Maximum seconds to wait for the LLM response
_TIMEOUT = 60

# ----------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """\
You are a post-processor for an Armenian cursive handwriting OCR system.
You receive raw, noisy OCR output in Armenian and must do TWO things:

1. **Fix errors** — correct spelling, grammar, punctuation, and word-boundary
   mistakes. Preserve the original meaning; do NOT paraphrase or add content.

2. **Classify content type** — decide whether the text represents:
   • "table" — the original image contained a table, grid, or structured
     columnar data (e.g. schedules, ledgers, lists with aligned columns).
   • "text"  — the original image contained free-form prose, paragraphs,
     a letter, notes, etc.

Return your answer as a JSON object with this EXACT schema (no markdown fences):

{
  "content_type": "table" | "text",
  "corrected_text": "<corrected full text>",
  "rows": [[...], ...]   // ONLY when content_type == "table"
}

Rules for the "rows" field:
• Include it ONLY when content_type is "table".
• Each inner list is one row; the first row should be column headers if present.
• Every cell value must be a string.

Rules for "corrected_text":
• Always include it regardless of content type.
• For tables, this is the corrected text as it appears naturally (before tabular parsing).

IMPORTANT: Return ONLY the raw JSON object. No markdown code fences, no explanation.
"""


# ----------------------------------------------------------------------- core

async def correct_and_classify(raw_text: str) -> Dict[str, Any]:
    """Send OCR text to OpenRouter and return corrected + classified result.

    Returns
    -------
    dict with keys:
        content_type : "table" | "text"
        corrected_text : str
        rows : list[list[str]]  (only when content_type == "table")
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set – returning raw text without correction")
        return {"content_type": "text", "corrected_text": raw_text}

    if not raw_text or not raw_text.strip():
        return {"content_type": "text", "corrected_text": ""}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://araks-ocr.app",
        "X-Title": "Araks Armenian OCR",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"OCR output:\n\n{raw_text}"},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            resp.raise_for_status()
    except httpx.TimeoutException:
        logger.error("OpenRouter request timed out after %ds", _TIMEOUT)
        return {"content_type": "text", "corrected_text": raw_text, "error": "LLM timeout"}
    except httpx.HTTPStatusError as exc:
        logger.error("OpenRouter HTTP %d: %s", exc.response.status_code, exc.response.text[:500])
        return {"content_type": "text", "corrected_text": raw_text,
                "error": f"LLM HTTP {exc.response.status_code}"}
    except httpx.HTTPError as exc:
        logger.error("OpenRouter request failed: %s", exc)
        return {"content_type": "text", "corrected_text": raw_text, "error": str(exc)}

    # Parse the LLM response
    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse OpenRouter response: %s", exc)
        return {"content_type": "text", "corrected_text": raw_text,
                "error": "Bad LLM response format"}

    return _parse_llm_output(content, raw_text)


def _parse_llm_output(content: str, fallback_text: str) -> Dict[str, Any]:
    """Parse the JSON that the LLM returns, with robust fallback."""
    # Strip markdown code fences if the model wraps them anyway
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON, using raw content as corrected text")
        return {"content_type": "text", "corrected_text": cleaned or fallback_text}

    # Validate expected keys
    content_type = data.get("content_type", "text")
    if content_type not in ("table", "text"):
        content_type = "text"

    corrected = data.get("corrected_text", fallback_text)

    result: Dict[str, Any] = {
        "content_type": content_type,
        "corrected_text": corrected,
    }

    if content_type == "table":
        rows = data.get("rows")
        if isinstance(rows, list) and all(isinstance(r, list) for r in rows):
            # Ensure every cell is a string
            result["rows"] = [[str(cell) for cell in row] for row in rows]
        else:
            # LLM said table but didn't give valid rows — fall back to text
            logger.warning("LLM classified as table but rows are invalid, falling back to text")
            result["content_type"] = "text"

    return result


async def correct_text_only(raw_text: str) -> str:
    """Convenience: just get corrected text, ignoring classification."""
    result = await correct_and_classify(raw_text)
    return result.get("corrected_text", raw_text)
