import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)

def extract_expiry(image_path):
    results = reader.readtext(image_path)
    pattern = r'(\d{2}[/-]\d{2}[/-]\d{2,4})'

    for bbox, text, conf in results:
        match = re.search(pattern, text)
        if match:
            return match.group(0), bbox, conf

    return None, None, 0.0