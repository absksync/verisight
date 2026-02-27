import easyocr
import re

reader = easyocr.Reader(["en"], gpu=False)


def extract_expiry(image_path):
    """
    Run OCR and try to extract an expiry-like date string plus its bbox and confidence.
    """
    results = reader.readtext(image_path)

    texts = [res[1] for res in results]
    bbox_list = [res[0] for res in results]
    conf_list = [res[2] for res in results]

    # Multiple expiry patterns, covering MM/YYYY, MM/YY, YYYY and DD/MM/YYYY variants.
    patterns = [
        r"\b(0[1-9]|1[0-2])[\/\-](20\d{2})\b",  # MM/YYYY
        r"\b(0[1-9]|1[0-2])[\/\-](\d{2})\b",  # MM/YY
        r"\b(20\d{2})\b",  # YYYY
        r"\b\d{2}[.\-/]\d{2}[.\-/]\d{2,4}\b",  # DD/MM/YYYY or DD-MM-YY etc.
        r"EXP[:\s]*(\d{2}[\/\-]\d{2,4})",  # EXP: 05/26 or 05/2026
    ]

    for i, text in enumerate(texts):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0), bbox_list[i], conf_list[i]

    return None, None, 0.0