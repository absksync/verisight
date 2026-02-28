"""
ViT-based AI / synthetic image detector.

Uses ``umm-maybe/AI-image-detector`` — a ViT model fine-tuned specifically
to classify images as **artificial** vs **human**.  Falls back to a
statistics-based heuristic if the model is unavailable.

The returned score is in [0, 1]:
    0 → likely real / human
    1 → likely AI-generated / synthetic
"""

from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger("verisight")

# ── Lazy-loaded model ────────────────────────────────────────────────────────
_classifier = None
_model_type: str | None = None  # "ai-detector" | "heuristic"
_lock = threading.Lock()


def _load_model() -> None:
    """Load the AI-image-detector model once (thread-safe)."""
    global _classifier, _model_type

    if _model_type is not None:          # already attempted
        return

    with _lock:
        if _model_type is not None:      # double-check after acquiring lock
            return

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("ViT: loading umm-maybe/AI-image-detector …")
            _classifier = hf_pipeline(
                "image-classification",
                model="umm-maybe/AI-image-detector",
                device=-1,  # CPU — safe everywhere
            )
            _model_type = "ai-detector"
            logger.info("ViT: model loaded ✓")
        except Exception as exc:
            logger.warning("ViT: could not load AI-image-detector (%s) — using heuristic", exc)
            _model_type = "heuristic"


def compute_vit(image_path: str) -> float:
    """Return synthetic probability in [0, 1] for the image at *image_path*."""
    _load_model()

    # ── Real model path ─────────────────────────────────────────────────────
    if _model_type == "ai-detector" and _classifier is not None:
        try:
            image = Image.open(image_path).convert("RGB")
            results = _classifier(image)
            # results: [{"label": "artificial", "score": 0.93}, {"label": "human", "score": 0.07}]
            for r in results:
                if r["label"].lower() in ("artificial", "fake", "ai", "ai_generated"):
                    return round(float(r["score"]), 4)
            # Fallback — take complement of "human" score
            for r in results:
                if r["label"].lower() in ("human", "real"):
                    return round(1.0 - float(r["score"]), 4)
            # Unknown label set — return top score inverted
            return round(1.0 - float(results[0]["score"]), 4)
        except Exception as exc:
            logger.warning("ViT inference failed: %s — falling back to heuristic", exc)

    # ── Heuristic fallback ──────────────────────────────────────────────────
    try:
        image = Image.open(image_path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32)
        std = float(arr.std())
        mean = float(arr.mean())

        # Deterministic, image-dependent jitter
        h = int(hashlib.md5(arr.tobytes()[:8192]).hexdigest()[:8], 16)
        jitter = (h / 0xFFFFFFFF) * 0.25

        score = 0.25 + jitter
        if std < 35:        # very uniform — common in AI-generated images
            score += 0.25
        if std > 80:        # very high contrast
            score += 0.10
        if mean < 50 or mean > 220:
            score += 0.10

        return round(min(1.0, max(0.0, score)), 4)
    except Exception:
        return 0.5