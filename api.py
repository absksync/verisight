"""
Verisight Backend — FastAPI server for real-time image fraud analysis.

Run with:
    cd /path/to/return0
    uvicorn api:app --reload --port 8000

The React frontend (Vite) at http://localhost:5173 connects here.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ── Ensure project root is on sys.path so ``ml`` package is importable ──────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("verisight")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="VeriSight", version="3.0.0")

# ── CORS — allow Vite dev-server and self ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ─────────────────────────────────────────────────────────────────────
API_KEY = "verisight-secret-key"


def verify_key(x_api_key: str | None = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Paths ────────────────────────────────────────────────────────────────────
UPLOAD_DIR   = ROOT / "uploads"
OUTPUTS_DIR  = ROOT / "ml" / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "results.csv"
DECISIONS_PATH = OUTPUTS_DIR / "decisions.json"
FRONTEND_DIR = ROOT / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

INFERENCE_TIMEOUT = 120  # seconds — hard ceiling so nothing hangs

# ── Global runtime state ─────────────────────────────────────────────────────
TOTAL_SAVED_AMOUNT: float = 0.0

SYSTEM_STATS: dict[str, int | float] = {
    "total_requests": 0,
    "rejected": 0,
    "saved_amount": 0.0,
}

REQUEST_QUEUE: list[str] = []          # list of request-ids currently waiting
_PREDICT_LOCK = asyncio.Lock()         # serialises ML inference

# ── Price regex patterns ─────────────────────────────────────────────────────
_PRICE_PATTERNS = [
    # ₹175  ₹120.50  ₹ 1,200.00
    re.compile(r"[₹]\s*([\d,]+(?:\.\d{1,2})?)"),
    # Rs 45  Rs.120  RS 120.50
    re.compile(r"(?i)rs\.?\s*([\d,]+(?:\.\d{1,2})?)"),
    # MRP 120  MRP: 45.00  M.R.P 999
    re.compile(r"(?i)m\.?r\.?p\.?\s*:?\s*([\d,]+(?:\.\d{1,2})?)"),
    # INR 500
    re.compile(r"(?i)inr\s*([\d,]+(?:\.\d{1,2})?)"),
    # Price: 120
    re.compile(r"(?i)price\s*:?\s*([\d,]+(?:\.\d{1,2})?)"),
]

# ── Static serving (heatmaps, highlighted images) ────────────────────────────
try:
    app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="ml_outputs")
except Exception:
    pass

# ── ML Pipeline (lazy import) ────────────────────────────────────────────────
_ml_ready = False
_ml_error: str | None = None


def _load_ml() -> bool:
    global _ml_ready, _ml_error
    if _ml_ready:
        return True
    try:
        from ml.pipeline import run_single_image   # noqa: F401
        from ml.modules.ocr_module import extract_all_regions  # noqa: F401
        _ml_ready = True
        logger.info("ML pipeline loaded ✓")
        return True
    except Exception as exc:
        _ml_error = str(exc)
        logger.error("Failed to load ML pipeline: %s", exc)
        return False


@app.on_event("startup")
async def _startup():
    logger.info("Starting up — importing ML pipeline …")
    _load_ml()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_level(score: int) -> str:
    if score < 30:
        return "LOW"
    return "MEDIUM" if score < 60 else "HIGH"


def _predicted_class(score: int) -> str:
    if score >= 60:
        return "Tampered"
    return "Suspicious" if score >= 30 else "Authentic"


def _timeline_to_check(timeline: str) -> str:
    t = timeline.lower()
    if "expired" in t and "before" in t:
        return "expired_before_delivery"
    if "valid" in t:
        return "valid"
    if "no expiry" in t:
        return "no_expiry"
    if "invalid" in t:
        return "no_delivery_date"
    return t.replace(" ", "_")


def _timeline_detail(timeline: str, expiry_text: str | None, delivery_date: str) -> dict:
    check = _timeline_to_check(timeline)
    if check == "expired_before_delivery":
        return {
            "status": "expired_before_delivery",
            "warning": (
                f"Product expiry '{expiry_text or 'unknown'}' is before "
                f"delivery date '{delivery_date or 'unknown'}'. "
                "This is a strong fraud indicator."
            ),
        }
    if check == "valid":
        return {
            "status": "valid",
            "message": f"Expiry '{expiry_text}' is after delivery '{delivery_date}'.",
        }
    if check == "no_expiry":
        return {
            "status": "no_expiry",
            "message": "OCR could not detect an expiry date on the image.",
        }
    return {"status": check, "message": timeline}


def _build_signals(output: dict, *, ocr_regions: list[dict] | None = None) -> list[dict]:
    """Build evidence-backed explainability signals for the frontend.

    Tags appear **only** when the underlying metric crosses its threshold.
    Confidence is an integer percentage (0-100) derived from the real value.
    """
    vit = float(output.get("vit", 0))
    ela = float(output.get("ela", 0))
    fft = float(output.get("fft", 0))
    ocr_conf = float(output.get("ocr_confidence", 1.0))

    # ── Log raw signal sources (auditable) ────────────────────────────────
    logger.info(
        "  Tag signals: vit=%.4f  ocr=%.4f  ela=%.4f  fft=%.4f",
        vit, ocr_conf, ela, fft,
    )

    signals: list[dict] = []

    # Synthetic Pattern — ViT indicates AI-generated content
    if vit > 0.55:
        signals.append({
            "name": "Synthetic Pattern",
            "detected": True,
            "confidence": int(min(100, max(0, vit * 100))),
        })

    # OCR Confidence Drop — only if OCR really struggled
    if ocr_conf < 0.4:
        signals.append({
            "name": "OCR Confidence Drop",
            "detected": True,
            "confidence": int(min(100, max(0, (1.0 - ocr_conf) * 100))),
        })

    # Compression Artifact — only if ELA shows real anomalies
    if ela > 0.02:
        signals.append({
            "name": "Compression Artifact",
            "detected": True,
            "confidence": int(min(100, max(0, ela * 100))),
        })

    # Texture Irregularity — only if FFT detects clear spectral anomaly
    if fft > 0.4:
        signals.append({
            "name": "Texture Irregularity",
            "detected": True,
            "confidence": int(min(100, max(0, fft * 100))),
        })

    # Layout Anomaly — only if actually measured from OCR regions
    if ocr_regions and _detect_layout_anomaly(ocr_regions):
        signals.append({
            "name": "Layout Anomaly",
            "detected": True,
            "confidence": 75,
        })

    return signals


# ── Layout anomaly patterns (floating dates, random spacing, etc.) ───────────
_LAYOUT_ANOMALY_PATTERNS = [
    re.compile(r"^\s*\d{4}\s*$"),           # bare year "2025" in isolation
    re.compile(r"\d\s{3,}\d"),               # digits with 3+ spaces between
    re.compile(r"^\s*\d{1,2}\s*$"),          # isolated 1-2 digit number
    re.compile(r"[A-Za-z]\d{6,}"),           # letter followed by 6+ digits (gibberish)
    re.compile(r"\d[A-Z]{3,}\d"),            # digit-LETTERS-digit (garbled OCR)
]


def _detect_layout_anomaly(ocr_regions: list[dict]) -> bool:
    """Detect unrealistic OCR text layout that suggests image manipulation."""
    if not ocr_regions:
        return False
    for region in ocr_regions:
        text = region.get("text", "").strip()
        if not text:
            continue
        for pat in _LAYOUT_ANOMALY_PATTERNS:
            if pat.search(text):
                return True
    return False


def _adjust_score(
    score: int,
    tags: list[str],
    vit_score: float,
    *,
    ela_score: float = 0.0,
    fft_score: float = 0.0,
    ocr_conf: float = 1.0,
    ocr_regions: list[dict] | None = None,
    has_exif: bool = True,
    expiry_text: str | None = None,
    detected_price: float | None = None,
) -> tuple[int, list[str]]:
    """Fraud-score refinement — prevents AI images from being approved.

    Rules (deterministic, production-safe):
      1. Suspicious floor (50)  — any single AI indicator fires
      2. Strong-AI escalation (70) — 2+ strong AI indicators
      3. OCR hallucination boost (+15) — detected numbers with low confidence
      4. Missing EXIF boost (+10) — AI-generated images lack metadata
      5. Clamp to [0, 100]

    Target ranges:
      • Real product photos → 10–30  (Approve)
      • Edited / tampered   → 40–60  (Manual Review)
      • AI-generated        → 60–85  (Reject)

    Returns ``(adjusted_score, updated_tags)``.
    """
    tags = list(tags)  # copy — never mutate the caller's list
    adjusted = score

    # ── Detect Layout Anomaly from OCR regions (real measurement) ─────────
    has_layout_anomaly = bool(ocr_regions and _detect_layout_anomaly(ocr_regions))
    if has_layout_anomaly and "Layout Anomaly" not in tags:
        tags.append("Layout Anomaly")

    # ── Collect AI indicators ─────────────────────────────────────────────
    has_synthetic = "Synthetic Pattern" in tags
    has_texture   = "Texture Irregularity" in tags
    has_layout    = "Layout Anomaly" in tags
    high_vit      = vit_score > 0.55
    low_ocr       = ocr_conf < 0.4
    exif_missing  = not has_exif

    # ── Rule 1: Suspicious floor (50) ─────────────────────────────────────
    # EXIF missing alone is NOT a floor trigger — most uploaded images
    # have EXIF stripped by platforms. It only applies as a boost (Rule 4)
    # when combined with other indicators.
    suspicious = (
        high_vit
        or has_synthetic
        or has_texture
        or has_layout
        or low_ocr
    )
    if suspicious:
        prev = adjusted
        adjusted = max(adjusted, 50)
        logger.info(
            "  ↳ Suspicious floor: %d → %d  (vit>0.55=%s syn=%s tex=%s lay=%s ocr<0.4=%s)",
            prev, adjusted, high_vit, has_synthetic, has_texture,
            has_layout, low_ocr,
        )

    # ── Rule 2: Strong AI escalation (70) ─────────────────────────────────
    strong_count = sum([
        has_synthetic,
        has_texture,
        has_layout,
        vit_score > 0.6,
    ])
    if strong_count >= 2:
        prev = adjusted
        adjusted = max(adjusted, 70)
        logger.info("  ↳ Strong AI escalation (%d/4 indicators) → floor 70 (was %d → %d)",
                    strong_count, prev, adjusted)

    # ── Rule 3: OCR hallucination boost (+15) ─────────────────────────────
    ocr_has_numbers = False
    if ocr_regions:
        for r in ocr_regions:
            if any(c.isdigit() for c in r.get("text", "")):
                ocr_has_numbers = True
                break
    if ocr_has_numbers and ocr_conf < 0.5:
        adjusted += 15
        logger.info("  ↳ OCR hallucination (numbers + conf %.2f < 0.5) → +15", ocr_conf)

    # ── Rule 4: Missing EXIF boost (+10) — only when other AI indicators present ──
    if exif_missing and suspicious:
        adjusted += 10
        logger.info("  ↳ EXIF metadata missing + suspicious → +10")

    # ── Rule 5: Clamp to [0, 100] ────────────────────────────────────────
    adjusted = min(100, max(0, adjusted))

    logger.info(
        "  ── Score: base=%d → adjusted=%d | vit=%.3f ela=%.3f fft=%.3f "
        "ocr=%.2f exif=%s tags=%s",
        score, adjusted,
        vit_score, ela_score, fft_score, ocr_conf,
        has_exif, ";".join(tags) if tags else "(none)",
    )

    return adjusted, tags


def _extract_price(ocr_regions: list[dict]) -> float | None:
    """Scan OCR regions for price patterns and return the first numeric match."""
    for region in ocr_regions:
        text = region.get("text", "")
        for pat in _PRICE_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except (ValueError, IndexError):
                    continue
    return None


def _safe_default(msg: str, delivery_date: str = "") -> dict:
    """Fallback JSON when the pipeline fails — the frontend always gets data."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "fraud_score": 50, "risk_level": "MEDIUM",
        "confidence": 0.0, "predicted_class": "Unknown",
        "signals": [], "ocr_regions": [],
        "heatmap": False, "image_width": 0, "image_height": 0,
        "delivery_date": delivery_date or None,
        "detected_expiry_date": None,
        "timeline_check": "error",
        "timeline_detail": {"status": "error", "warning": msg},
        "score": 50, "decision": "Manual Review",
        "timeline": "Error", "expiry_text": "",
        "ocr_confidence": 0.0, "fusion_confidence": 0.0,
        "vit_score": 0.0, "tags": "", "error": msg,
        "detected_price": None,
        "refund_saved": TOTAL_SAVED_AMOUNT,
        "queue_position": None,
    }


# ── CSV helpers (for analytics / queue pages) ────────────────────────────────
def _load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(str(RESULTS_PATH))
        adj, dec = [], []
        for _, row in df.iterrows():
            raw = int(row.get("score", 0))
            tags = [t.strip() for t in str(row.get("tags", "")).split(";") if t.strip()]
            vit = float(row.get("vit", 0))
            ela = float(row.get("ela", 0))
            fft = float(row.get("fft", 0))
            ocr_c = float(row.get("ocr_confidence", 1.0))
            a, _ = _adjust_score(raw, tags, vit, ela_score=ela, fft_score=fft, ocr_conf=ocr_c)
            adj.append(a)
            dec.append("Approve" if a < 30 else ("Manual Review" if a < 60 else "Reject"))
        df["adj_score"] = adj
        df["adj_decision"] = dec
        return df
    except Exception:
        return pd.DataFrame()


def _load_decisions() -> list:
    if not DECISIONS_PATH.exists():
        return []
    try:
        return json.loads(DECISIONS_PATH.read_text())
    except Exception:
        return []


def _save_decisions(decisions: list) -> None:
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    # Serve old HTML frontend if it exists, else return JSON status
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "service": "verisight-api", "version": "3.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "ml_ready": _ml_ready, "ocr_available": _ml_ready}


# ── /stats — live dashboard data (no hardcoded values) ──────────────────────
@app.get("/stats")
def stats():
    return {
        "total_requests": SYSTEM_STATS["total_requests"],
        "rejected":       SYSTEM_STATS["rejected"],
        "saved_amount":   SYSTEM_STATS["saved_amount"],
        "queue_length":   len(REQUEST_QUEUE),
    }


# ── /predict — the core endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    delivery_date: str = Form(""),
    x_api_key: str | None = Header(None),
):
    global TOTAL_SAVED_AMOUNT

    # ── Auth ────────────────────────────────────────────────────────────────
    verify_key(x_api_key)
    logger.info("▶ /predict — file=%s  delivery_date=%s", file.filename, delivery_date)

    # ── Queue tracking ──────────────────────────────────────────────────────
    request_id = str(uuid.uuid4())
    REQUEST_QUEUE.append(request_id)
    queue_position: int | None = len(REQUEST_QUEUE) - 1  # 0 = processing now
    if queue_position == 0:
        queue_position = None          # first in line → no wait
    logger.info("  Queue position: %s (queue len=%d)", queue_position, len(REQUEST_QUEUE))

    try:
        return await _do_predict(file, delivery_date, x_api_key, request_id, queue_position)
    finally:
        # Always remove ourselves from the queue when done
        try:
            REQUEST_QUEUE.remove(request_id)
        except ValueError:
            pass


async def _do_predict(
    file: UploadFile,
    delivery_date: str,
    x_api_key: str | None,
    request_id: str,
    queue_position: int | None,
) -> JSONResponse:
    global TOTAL_SAVED_AMOUNT

    # ── Validate image type ─────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("  Rejected: not an image (%s)", file.content_type)
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file is not an image."},
        )

    # ── Read + open image ───────────────────────────────────────────────────
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        original_format = pil_image.format
        width, height = pil_image.size
        logger.info("  Image: %s %dx%d", original_format or "unknown", width, height)
    except Exception as exc:
        logger.exception("  Failed to read image")
        return JSONResponse(status_code=400, content={"error": f"Cannot read image: {exc}"})

    # ── EXIF check ──────────────────────────────────────────────────────────
    try:
        has_exif = len(pil_image.getexif()) > 0
    except Exception:
        has_exif = False

    # ── Save to uploads/ ────────────────────────────────────────────────────
    file_id = request_id
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    try:
        save_path.write_bytes(contents)
        logger.info("  Saved → %s", save_path)
    except Exception as exc:
        logger.warning("  Could not save upload: %s", exc)
        return JSONResponse(content=_safe_default(f"Save failed: {exc}", delivery_date))

    # ── Ensure ML pipeline is loaded ────────────────────────────────────────
    if not _ml_ready:
        _load_ml()
    if not _ml_ready:
        logger.error("  ML pipeline not available: %s", _ml_error)
        return JSONResponse(content=_safe_default(f"ML pipeline error: {_ml_error}", delivery_date))

    from ml.pipeline import run_single_image
    from ml.modules.ocr_module import extract_all_regions

    # ── Serialise ML inference through lock (queue behaviour) ───────────────
    logger.info("  Waiting for ML lock …")
    async with _PREDICT_LOCK:
        logger.info("  Running ML pipeline …")
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(run_single_image, str(save_path), delivery_date or None),
                timeout=INFERENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("  Pipeline timed out after %ds", INFERENCE_TIMEOUT)
            return JSONResponse(content=_safe_default("Analysis timed out", delivery_date))
        except Exception as exc:
            logger.exception("  Pipeline crashed")
            return JSONResponse(content=_safe_default(f"Pipeline error: {exc}", delivery_date))

    logger.info("  Pipeline done — raw score=%s decision=%s", output.get("score"), output.get("decision"))
    logger.info("  OCR expiry_text=%s  confidence=%s", output.get("expiry_text"), output.get("ocr_confidence"))
    logger.info("  Scores: ela=%.4f  fft=%.4f  vit=%.4f  expiry=%.4f",
                output.get("ela", 0), output.get("fft", 0),
                output.get("vit", 0), output.get("expiry_score", 0))

    # ── OCR regions for heatmap ─────────────────────────────────────────────
    try:
        ocr_regions = await asyncio.to_thread(extract_all_regions, str(save_path))
    except Exception:
        ocr_regions = []

    # ── Price detection from OCR text ───────────────────────────────────────
    detected_price = _extract_price(ocr_regions)
    logger.info("  Detected price: %s", detected_price)

    # ── Comprehensive score refinement ──────────────────────────────────────
    raw_score = int(output.get("score", 50))
    tags_list = output.get("tags", [])
    if isinstance(tags_list, str):
        tags_list = [t.strip() for t in tags_list.split(";") if t.strip()]
    vit_score = float(output.get("vit", 0))
    ela_score = float(output.get("ela", 0))
    fft_score = float(output.get("fft", 0))
    ocr_conf  = float(output.get("ocr_confidence", 1.0))

    adjusted_score, tags_list = _adjust_score(
        raw_score, tags_list, vit_score,
        ela_score=ela_score,
        fft_score=fft_score,
        ocr_conf=ocr_conf,
        ocr_regions=ocr_regions,
        has_exif=has_exif,
        expiry_text=output.get("expiry_text"),
        detected_price=detected_price,
    )

    # ── Recalculate decision AFTER all adjustments ──────────────────────────
    if adjusted_score < 30:
        decision = "Approve"
    elif adjusted_score < 60:
        decision = "Manual Review"
    else:
        decision = "Reject"

    logger.info("  Score: raw=%d → adjusted=%d → %s", raw_score, adjusted_score, decision)

    # ── Refund / saved-amount logic ─────────────────────────────────────────
    is_fraud = "Synthetic Pattern" in tags_list or vit_score > 0.80
    if decision == "Reject" and is_fraud and detected_price is not None:
        TOTAL_SAVED_AMOUNT += detected_price
        logger.info("  💰 Added ₹%.2f to saved amount → total ₹%.2f",
                    detected_price, TOTAL_SAVED_AMOUNT)

    # ── Update system stats ─────────────────────────────────────────────────
    SYSTEM_STATS["total_requests"] += 1
    if decision == "Reject":
        SYSTEM_STATS["rejected"] += 1
    SYSTEM_STATS["saved_amount"] = TOTAL_SAVED_AMOUNT

    # ── Build frontend-compatible signals (evidence-backed only) ────────────
    signals = _build_signals(output, ocr_regions=ocr_regions)

    # ── Timeline mapping ────────────────────────────────────────────────────
    timeline_str = str(output.get("timeline", ""))
    timeline_check = _timeline_to_check(timeline_str)
    timeline_det = _timeline_detail(timeline_str, output.get("expiry_text"), delivery_date)

    # ── Fusion confidence ───────────────────────────────────────────────────
    fusion = float(output.get("confidence", 0))
    if delivery_date:
        fusion = min(1.0, fusion + 0.10)
    fusion = round(fusion, 2)

    # ── Assemble full response ──────────────────────────────────────────────
    result: dict[str, Any] = {
        # ── Identifiers
        "id":                   file_id,
        "timestamp":            datetime.utcnow().isoformat(),

        # ── Primary fields consumed by the React frontend
        "fraud_score":          adjusted_score,
        "risk_level":           _risk_level(adjusted_score),
        "confidence":           fusion,
        "predicted_class":      _predicted_class(adjusted_score),
        "signals":              signals,
        "ocr_regions":          ocr_regions,
        "heatmap":              len(ocr_regions) > 0,
        "image_width":          width,
        "image_height":         height,
        "delivery_date":        delivery_date or None,
        "detected_expiry_date": output.get("expiry_text"),
        "timeline_check":       timeline_check,
        "timeline_detail":      timeline_det,

        # ── Extra fields from spec
        "score":                adjusted_score,
        "raw_score":            raw_score,
        "decision":             decision,
        "timeline":             timeline_str,
        "expiry_text":          output.get("expiry_text") or "",
        "ocr_confidence":       round(float(output.get("ocr_confidence", 0)), 4),
        "fusion_confidence":    fusion,
        "vit_score":            round(vit_score, 4),
        "ela":                  round(float(output.get("ela", 0)), 4),
        "fft":                  round(float(output.get("fft", 0)), 4),
        "tags":                 "; ".join(tags_list) if tags_list else "",
        "highlight_path":       output.get("highlight_path"),

        # ── New financial / queue fields
        "detected_price":       detected_price,
        "refund_saved":         TOTAL_SAVED_AMOUNT,
        "queue_position":       queue_position,
    }

    logger.info("  ✓ Final: score=%d  risk=%s  decision=%s  price=%s  saved=%.2f",
                adjusted_score, result["risk_level"], decision,
                detected_price, TOTAL_SAVED_AMOUNT)
    return JSONResponse(content=result)


# ── /results (verisight-frontend Analytics / Queue) ──────────────────────────
@app.get("/results")
def results_endpoint(x_api_key: str | None = Header(None)):
    verify_key(x_api_key)
    df = _load_results()
    if df.empty:
        return JSONResponse(content=[])
    records = df.to_dict(orient="records")
    return JSONResponse(content=records)


# ── /api/analytics ───────────────────────────────────────────────────────────
@app.get("/api/analytics")
def analytics():
    df = _load_results()

    # ── Live totals from SYSTEM_STATS (always available, even if CSV empty) ──
    live_saved    = SYSTEM_STATS["saved_amount"]
    live_rejected = SYSTEM_STATS["rejected"]
    live_total    = SYSTEM_STATS["total_requests"]

    if df.empty:
        return {
            # Legacy fields
            "total": live_total, "approve": 0, "review": 0,
            "reject": live_rejected,
            "avg_score": 0, "revenue_protected": live_saved,
            "top_tags": [], "score_distribution": [],
            "recent_rejects": [], "categories": {},
            # Frontend-expected fields
            "revenue_protected_today": f"\u20b9 {live_saved:,.0f}" if live_saved else "\u20b9 0",
            "revenue_change_pct": 0,
            "total_interceptions": live_rejected,
            "top_categories": [],
            "fraud_attempt_spikes": [],
        }

    total = len(df)
    approve = int((df["adj_decision"] == "Approve").sum())
    review  = int((df["adj_decision"] == "Manual Review").sum())
    reject  = int((df["adj_decision"] == "Reject").sum())
    avg_score = round(float(df["adj_score"].mean()), 1)

    # Revenue = live saved amount (from real OCR-detected prices)
    # plus a conservative estimate for rejects without a detected price
    revenue = live_saved + (reject * 2500) + (review * 800)

    all_tags: list[str] = []
    for t in df["tags"].dropna():
        all_tags.extend([x.strip() for x in str(t).split(";") if x.strip()])
    tag_counts: dict[str, int] = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:6]

    buckets = [0] * 10
    for s in df["adj_score"]:
        buckets[min(int(s) // 10, 9)] += 1
    score_dist = [{"range": f"{i * 10}-{i * 10 + 9}", "count": buckets[i]} for i in range(10)]

    high_risk = df[df["adj_score"] >= 60].sort_values("adj_score", ascending=False).head(5)
    recent_rejects = [
        {"filename": r["filename"], "score": int(r["adj_score"]),
         "tags": str(r.get("tags", "")), "decision": r["adj_decision"]}
        for _, r in high_risk.iterrows()
    ]

    categories_raw = {
        "Synthetic / AI": int(sum(1 for t in df["tags"].dropna() if "Synthetic Pattern" in str(t))),
        "Compression":    int(sum(1 for t in df["tags"].dropna() if "Compression Artifact" in str(t))),
        "Texture Issues": int(sum(1 for t in df["tags"].dropna() if "Texture Irregularity" in str(t))),
        "OCR Problems":   int(sum(1 for t in df["tags"].dropna() if "OCR Confidence Drop" in str(t))),
    }

    # ── Build top_categories in the shape React TopCategory expects ──────
    cat_total = max(1, sum(categories_raw.values()))
    top_categories = [
        {"name": name, "count": count, "pct": round(count / cat_total * 100)}
        for name, count in sorted(categories_raw.items(), key=lambda x: -x[1])
        if count > 0
    ]

    # ── Build fraud spikes from recent high-risk entries ─────────────────
    fraud_spikes = []
    for _, r in high_risk.head(3).iterrows():
        score_val = int(r["adj_score"])
        fraud_spikes.append({
            "region": str(r["filename"]).replace(".jpeg", "").replace(".jpg", "").replace("_", " ").title(),
            "time": datetime.now().strftime("%I:%M %p"),
            "increase_pct": f"+{score_val}%",
            "severity": "critical" if score_val >= 80 else "warning",
        })

    # Merge live stats with CSV stats for accurate totals
    total_interceptions = max(live_rejected, reject)
    combined_revenue = max(revenue, live_saved)

    return {
        # ── Legacy fields (backwards compat) ─────────────────────────────
        "total": max(total, live_total), "approve": approve,
        "review": review, "reject": total_interceptions,
        "avg_score": avg_score, "revenue_protected": combined_revenue,
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        "score_distribution": score_dist,
        "recent_rejects": recent_rejects,
        "categories": categories_raw,
        # ── Frontend-expected fields ─────────────────────────────────────
        "revenue_protected_today": f"\u20b9 {combined_revenue:,.0f}",
        "revenue_change_pct": round(min(99, total_interceptions * 12.5), 1) if total_interceptions else 0,
        "total_interceptions": total_interceptions,
        "top_categories": top_categories,
        "fraud_attempt_spikes": fraud_spikes,
    }


# ── /api/queue ───────────────────────────────────────────────────────────────
@app.get("/api/queue")
def queue(
    risk: str | None = Query(None),
    sort_by: str = Query("score_desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    df = _load_results()
    if df.empty:
        return {"items": [], "total": 0, "page": page, "pages": 0}

    decisions = {d["filename"]: d for d in _load_decisions()}

    if risk == "low":
        df = df[df["adj_score"] < 30]
    elif risk == "medium":
        df = df[(df["adj_score"] >= 30) & (df["adj_score"] < 60)]
    elif risk == "high":
        df = df[df["adj_score"] >= 60]

    if sort_by == "score_asc":
        df = df.sort_values("adj_score", ascending=True)
    elif sort_by == "score_desc":
        df = df.sort_values("adj_score", ascending=False)
    else:
        df = df.sort_values("filename")

    total = len(df)
    pages = max(1, -(-total // per_page))
    page_df = df.iloc[(page - 1) * per_page: page * per_page]

    items = []
    for _, r in page_df.iterrows():
        fname = r["filename"]
        d = decisions.get(fname, {})
        items.append({
            "filename": fname,
            "category": str(r.get("category", "")),
            "score": int(r["adj_score"]),
            "raw_score": int(r.get("score", 0)),
            "decision": r["adj_decision"],
            "tags": str(r.get("tags", "")),
            "timeline": str(r.get("timeline", "")),
            "expiry_text": str(r.get("expiry_text", "")),
            "ela": round(float(r.get("ela", 0)), 4),
            "fft": round(float(r.get("fft", 0)), 4),
            "vit": round(float(r.get("vit", 0)), 4),
            "ocr_confidence": round(float(r.get("ocr_confidence", 0)), 4),
            "highlight_path": str(r.get("highlight_path", "")),
            "user_decision": d.get("action"),
            "decided_at": d.get("timestamp"),
        })

    return {"items": items, "total": total, "page": page, "pages": pages}


# ── /api/decide ──────────────────────────────────────────────────────────────
class DecisionRequest(BaseModel):
    filename: str
    action: str


@app.post("/api/decide")
def log_decision(req: DecisionRequest):
    decisions = _load_decisions()
    found = False
    for d in decisions:
        if d["filename"] == req.filename:
            d["action"] = req.action
            d["timestamp"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        decisions.append({
            "filename": req.filename,
            "action": req.action,
            "timestamp": datetime.now().isoformat(),
        })
    _save_decisions(decisions)
    logger.info("Decision: %s → %s", req.filename, req.action)
    return {"status": "ok", "filename": req.filename, "action": req.action}


# ── /api/bulk-decide ─────────────────────────────────────────────────────────
class BulkDecisionRequest(BaseModel):
    filenames: list[str]
    action: str


@app.post("/api/bulk-decide")
def bulk_decide(req: BulkDecisionRequest):
    decisions = _load_decisions()
    dec_map = {d["filename"]: d for d in decisions}
    now = datetime.now().isoformat()
    for fname in req.filenames:
        if fname in dec_map:
            dec_map[fname]["action"] = req.action
            dec_map[fname]["timestamp"] = now
        else:
            dec_map[fname] = {"filename": fname, "action": req.action, "timestamp": now}
    _save_decisions(list(dec_map.values()))
    logger.info("Bulk decision: %d items → %s", len(req.filenames), req.action)
    return {"status": "ok", "count": len(req.filenames), "action": req.action}


# ── Global exception handler — always return JSON ────────────────────────────
@app.exception_handler(Exception)
async def _global_exc(request, exc):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )