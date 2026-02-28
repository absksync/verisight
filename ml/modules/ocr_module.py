"""
OCR module — extracts text from images using EasyOCR.

Provides:
    extract_expiry(image_path)      → (expiry_text, bbox, confidence)
    extract_all_regions(image_path) → list of {text, confidence, bbox}

The EasyOCR reader is lazily loaded on first use and results are cached
per image path so that calling both functions on the same file does NOT
trigger two OCR runs.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger("verisight")

# ── Lazy reader ──────────────────────────────────────────────────────────────
_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    """Return the shared EasyOCR reader (created once)."""
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is not None:
            return _reader
        try:
            import easyocr
            logger.info("OCR: loading EasyOCR reader …")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("OCR: reader loaded ✓")
        except Exception as exc:
            logger.error("OCR: failed to load EasyOCR: %s", exc)
            _reader = None
    return _reader


# ── Result cache (avoids double OCR on same image) ──────────────────────────
_ocr_cache: dict[str, list] = {}
_CACHE_MAX = 50


def _run_ocr_raw(image_path: str) -> list:
    """Run EasyOCR on *image_path* and return raw result list (cached)."""
    key = str(image_path)
    if key in _ocr_cache:
        return _ocr_cache[key]

    reader = _get_reader()
    if reader is None:
        return []

    try:
        results = reader.readtext(str(image_path))
    except Exception as exc:
        logger.error("OCR readtext failed: %s", exc)
        return []

    _ocr_cache[key] = results
    # Evict oldest entry if cache grows too large
    if len(_ocr_cache) > _CACHE_MAX:
        _ocr_cache.pop(next(iter(_ocr_cache)), None)

    return results


def clear_cache() -> None:
    """Explicitly clear the OCR cache (useful between requests)."""
    _ocr_cache.clear()


# ── Expiry extraction ────────────────────────────────────────────────────────
# Each pattern must have exactly ONE top-level capturing group that extracts
# *only* the date portion (not the keyword prefix).
_EXPIRY_PATTERNS = [
    # Patterns WITH keyword prefix (higher priority)
    r"(?:EXP(?:IRY|IRES?)?|BEST\s*BEFORE|BB|USE\s*BY)[:\s./-]*(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})",    # EXP YYYY-MM-DD
    r"(?:EXP(?:IRY|IRES?)?|BEST\s*BEFORE|BB|USE\s*BY)[:\s./-]*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",    # EXP DD/MM/YYYY
    r"(?:EXP(?:IRY|IRES?)?|BEST\s*BEFORE|BB|USE\s*BY)[:\s./-]*(\d{1,2}[.\-/]\d{2,4})",                   # EXP MM/YYYY

    # Bare date formats (fallback)
    r"\b(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})\b",           # YYYY-MM-DD or YYYY/MM/DD
    r"\b((?:0[1-9]|1[0-2])[/\-]20\d{2})\b",                # MM/YYYY
    r"\b((?:0[1-9]|1[0-2])[/\-]\d{2})\b",                  # MM/YY
    r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})\b",               # DD/MM/YYYY or DD-MM-YY
    r"\b(20\d{2})\b",                                        # YYYY alone
]


def extract_expiry(image_path: str) -> tuple[str | None, Any, float]:
    """
    Run OCR and try to extract an expiry-like date string.

    Returns:
        (expiry_text, bbox, confidence)
    """
    results = _run_ocr_raw(image_path)
    if not results:
        return None, None, 0.0

    for bbox, text, conf in results:
        for pattern in _EXPIRY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Always return group(1) — the date portion only
                return match.group(1), bbox, float(conf)

    return None, None, 0.0


# ── All-regions extraction (for heatmap overlay) ────────────────────────────
def extract_all_regions(image_path: str) -> list[dict]:
    """
    Return every OCR-detected text region in frontend-compatible format::

        [{"text": "...", "confidence": 0.92, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
    """
    results = _run_ocr_raw(image_path)
    regions: list[dict] = []
    for bbox, text, conf in results:
        regions.append({
            "text": text,
            "confidence": round(float(conf), 2),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })
    return regions