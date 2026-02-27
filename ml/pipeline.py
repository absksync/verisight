from .modules.ocr_module import extract_expiry
from .modules.expiry_logic import check_expiry
from .modules.ela_module import compute_ela
from .modules.fft_module import compute_fft
from .modules.vit_module import compute_vit
from .fusion import compute_score
from .tags_engine import generate_tags
from .confidence_engine import compute_confidence
from .utils.image_utils import highlight_region


def run_pipeline(image_path, delivery_date):
    """
    End-to-end authenticity pipeline for a single claim image.

    1. OCR: read expiry text and confidence from packaging.
    2. Context check: compare expiry date against delivery date.
    3. Forensics: ELA, FFT and ViT based cues.
    4. Fusion: combine all signals into a 0–100 risk score and decision.
    """

    # ---- OCR & context-aware expiry logic ----
    expiry_text, bbox, ocr_conf = extract_expiry(image_path)
    timeline_msg, expiry_score = check_expiry(expiry_text, delivery_date)

    # ---- Forensic Analysis (image manipulation / synthesis) ----
    ela_score = compute_ela(image_path)
    fft_score = compute_fft(image_path)
    vit_score = compute_vit(image_path)

    # ---- Final Risk Score (0–100, higher = more suspicious) ----
    # Using a simple, interpretable weighted fusion of all four signals.
    final_score = compute_score(
        ela_score,
        fft_score,
        vit_score,
        expiry_score,
    )

    # ---- Decision Thresholds ----
    #   - Approve: 0–35         (low overall risk)
    #   - Manual Review: 35–80  (uncertain, human check)
    #   - Reject: 80+           (only extremely high risk)
    # The high Reject threshold is intentional to avoid false positives;
    # on the current dataset almost all suspicious cases are sent to
    # manual review rather than auto-rejected.
    if final_score < 30:
        decision = "Approve"
    elif final_score < 45:
        decision = "Manual Review"
    else:
        decision = "Reject"
    # ---- Confidence & explanatory tags ----
    fusion_confidence = compute_confidence(
        [ela_score, fft_score, vit_score, expiry_score]
    )
    tags = generate_tags(ela_score, fft_score, vit_score, ocr_conf)
    highlighted_path = highlight_region(image_path, bbox)

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
        "tags": tags,
        "highlight_path": highlighted_path,
    }