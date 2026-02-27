def generate_tags(ela, fft, vit, ocr_conf):
    tags = []

    if ela > 0.6:
        tags.append("Compression Artifact")

    if fft > 0.5:
        tags.append("Texture Irregularity")

    if vit > 0.6:
        tags.append("Synthetic Pattern")

    if ocr_conf < 0.6:
        tags.append("OCR Confidence Drop")

    return tags