def generate_tags(ela, fft, vit, ocr_conf):
    """Generate explainability tags ONLY when backed by real measurements.

    Each tag is a plain string.  The companion ``build_evidence_signals()``
    in api.py converts them to ``{"name", "confidence"}`` dicts with
    normalised integer percentages for the frontend.

    Thresholds (user-spec):
      • Synthetic Pattern    — vit  > 0.8
      • Compression Artifact — ela  > 0.02
      • Texture Irregularity — fft  > 0.3
      • OCR Confidence Drop  — ocr  < 0.4
    """
    tags = []

    if vit > 0.8:
        tags.append("Synthetic Pattern")

    if ocr_conf < 0.4:
        tags.append("OCR Confidence Drop")

    if ela > 0.02:
        tags.append("Compression Artifact")

    if fft > 0.3:
        tags.append("Texture Irregularity")

    return tags