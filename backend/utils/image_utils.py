import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Dict, Any, List, Optional, Tuple

def pil_to_base64(pil_img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL Image to a base64 data URI."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG":
        pil_img.convert("RGB").save(buffered, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        pil_img.save(buffered, format="PNG")
        mime = "image/png"
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{img_str}"

def cv2_to_base64(cv_img: np.ndarray, format: str = ".jpg", quality: int = 85) -> str:
    """Converts an OpenCV BGR image to a base64 data URI."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if format.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(format, cv_img, encode_params)
    if not success:
        raise ValueError("Failed to encode OpenCV image")
    mime = "image/jpeg" if format.lower() in [".jpg", ".jpeg"] else "image/png"
    img_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{img_str}"

def base64_to_cv2(b64_str: str) -> np.ndarray:
    """Decodes a base64 data URI to an OpenCV BGR image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv

def draw_annotations(
    orig_img_bgr: np.ndarray,
    date_info: Dict[str, Any],
    tamper_info: Dict[str, Any],
    ocr_tokens: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Draws high-contrast bounding boxes, badges, and labels on the image for:
    - Expiration Date (Green if VALID, Orange if EXPIRING_SOON, Red if EXPIRED)
    - Manufacturing Date (Blue)
    - Batch/Lot (Purple)
    - Tampered Regions (Dotted/Highlighted Red)
    """
    annotated = orig_img_bgr.copy()
    h, w = annotated.shape[:2]

    # Dynamic line thickness based on image resolution
    thickness = max(2, int(min(w, h) / 350))
    font_scale = max(0.45, min(w, h) / 1000.0)

    # 1. Color mapping based on expiry status
    status = date_info.get("expiry_status", "UNKNOWN")
    if status == "VALID":
        exp_color = (46, 204, 113)  # Vibrant Emerald Green
    elif status == "EXPIRING_SOON":
        exp_color = (0, 165, 255)   # Amber Orange
    elif status == "EXPIRED":
        exp_color = (34, 34, 235)   # Bright Red
    else:
        exp_color = (200, 200, 200) # Neutral Gray

    mfg_color = (255, 165, 0)     # Sky/Cyan Blue
    batch_color = (204, 50, 153)  # Magenta/Purple
    tamper_color = (0, 0, 255)    # Danger Red

    # Helper to draw a modern pill badge
    def draw_badge(img, label, x, y, bg_color):
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        pad = 4
        y_top = max(0, y - label_h - pad * 2)
        cv2.rectangle(
            img,
            (x, y_top),
            (x + label_w + pad * 2, y_top + label_h + pad * 2),
            bg_color,
            cv2.FILLED
        )
        cv2.putText(
            img,
            label,
            (x + pad, y_top + label_h + pad - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Draw Tampered Anomalous Regions first
    for reg in tamper_info.get("anomalous_regions", []):
        x1, y1, x2, y2 = reg["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), tamper_color, thickness)
        draw_badge(annotated, f"TAMPER ALERT ({int(reg['score']*100)}%)", x1, y1, tamper_color)

    # Draw Expiry Box
    if date_info.get("expiry_bbox"):
        x1, y1, x2, y2 = date_info["expiry_bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), exp_color, thickness + 1)
        exp_label = f"EXP: {date_info.get('expiry_date', 'N/A')} [{status}]"
        draw_badge(annotated, exp_label, x1, y1, exp_color)

    # Draw MFG Box
    if date_info.get("mfg_bbox"):
        x1, y1, x2, y2 = date_info["mfg_bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), mfg_color, thickness)
        mfg_label = f"MFG: {date_info.get('mfg_date', 'N/A')}"
        draw_badge(annotated, mfg_label, x1, y1, mfg_color)

    # Draw Batch Box
    if date_info.get("batch_bbox"):
        x1, y1, x2, y2 = date_info["batch_bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), batch_color, thickness)
        batch_label = f"BATCH: {date_info.get('batch_number')}"
        draw_badge(annotated, batch_label, x1, y1, batch_color)

    return annotated
