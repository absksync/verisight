"""
Verisight ML Pipeline — single-image fraud analysis.

    run_pipeline(image_path, delivery_date)       → dict (may raise)
    run_single_image(image_path, delivery_date)   → dict (never raises)
"""

from __future__ import annotations

import logging

from .modules.ocr_module import extract_expiry, clear_cache
from .modules.expiry_logic import check_expiry
from .modules.ela_module import compute_ela
from .modules.fft_module import compute_fft
from .modules.vit_module import compute_vit
from .fusion import compute_score
from .tags_engine import generate_tags
from .confidence_engine import compute_confidence
from .utils.image_utils import highlight_region

logger = logging.getLogger("verisight")


def run_pipeline(image_path: str, delivery_date: str | None) -> dict:
    """
    End-to-end authenticity pipeline for a single claim image.

    1. OCR: read expiry text and confidence from packaging.
    2. Context check: compare expiry date against delivery date.
    3. Forensics: ELA, FFT and ViT based cues.
    4. Fusion: combine all signals into a 0-100 risk score and decision.
    """

    # ── OCR & context-aware expiry logic ────────────────────────────────────
    expiry_text, ocr_bbox, ocr_conf = extract_expiry(image_path)
    timeline_msg, expiry_score = check_expiry(expiry_text, delivery_date)

    # ── Forensic Analysis (image manipulation / synthesis) ──────────────────
    ela_score = compute_ela(image_path)
    fft_score = compute_fft(image_path)
    vit_score = compute_vit(image_path)

    # ── Final Risk Score (0-100, higher = more suspicious) ──────────────────
    final_score = compute_score(
        ela_score,
        fft_score,
        vit_score,
        expiry_score,
    )

    # ── Decision Thresholds ─────────────────────────────────────────────────
    if final_score < 30:
        decision = "Approve"
    elif final_score < 60:
        decision = "Manual Review"
    else:
        decision = "Reject"

    # ── Confidence & explanatory tags ───────────────────────────────────────
    fusion_confidence = compute_confidence(
        [ela_score, fft_score, vit_score, expiry_score]
    )
    tags = generate_tags(ela_score, fft_score, vit_score, ocr_conf)
    highlighted_path = highlight_region(image_path, ocr_bbox)

    return {
        "score": final_score,
        "decision": decision,
        "confidence": fusion_confidence,
        "expiry_text": expiry_text,
        "timeline": timeline_msg,
        "ela": ela_score,
        "fft": fft_score,
        "vit": vit_score,
        "expiry_score": expiry_score,
        "ocr_confidence": ocr_conf,
        "ocr_bbox": ocr_bbox,
        "tags": tags,
        "highlight_path": highlighted_path,
    }


def run_single_image(image_path: str, delivery_date: str | None) -> dict:
    """
    Safe wrapper around :func:`run_pipeline`.

    * Clears OCR cache so a fresh upload always gets a fresh OCR run.
    * NEVER raises — returns safe defaults on any failure.
    """
    # Clear OCR cache so this upload is analyzed fresh
    clear_cache()

    try:
        result = run_pipeline(image_path, delivery_date)
        logger.info("Pipeline OK — score=%s", result.get("score"))
        return result
    except Exception as exc:
        logger.exception("Pipeline crashed: %s", exc)
        return {
            "score": 50,
            "decision": "Manual Review",
            "confidence": 0.0,
            "expiry_text": None,
            "timeline": "Analysis error",
            "ela": 0.0,
            "fft": 0.0,
            "vit": 0.0,
            "expiry_score": 0.0,
            "ocr_confidence": 0.0,
            "ocr_bbox": None,
            "tags": ["Analysis Error"],
            "highlight_path": None,
        }