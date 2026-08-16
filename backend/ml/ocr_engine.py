import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging
from typing import Dict, Any, List, Tuple, Optional
import os
import re

logger = logging.getLogger(__name__)

if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

class OCREngine:
    """
    Advanced Industrial Packaging OCR Engine.
    Employs Morphological Region Proposals, Multi-Scale Contrast Enhancement,
    and Dot-Matrix / Thermal Inkjet Reconstruction to accurately locate
    and extract dates and text from complex commercial packaging.
    """

    def __init__(self):
        self.tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract check failed: {e}")
            return False

    def generate_candidate_regions(self, cv_img: np.ndarray) -> List[List[int]]:
        """
        Proposes bounding box regions likely to contain text, date stamps,
        or batch codes using directional morphological gradient saliency.
        """
        h, w = cv_img.shape[:2]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)

        proposed_boxes = []

        # Multiple kernel scales to detect both compact date stamps and multi-line nutritional info
        kernel_widths = [20, 40, 70, 110]
        for kw in kernel_widths:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 7))
            closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
            _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                aspect = bw / float(bh) if bh > 0 else 0

                # Filter valid text shapes
                if 12 <= bh <= (h * 0.4) and 30 <= bw <= (w * 0.95) and (w * h * 0.001) < area < (w * h * 0.35):
                    # Add context padding around the crop
                    pad_x = int(bw * 0.08)
                    pad_y = int(bh * 0.12)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + bw + pad_x)
                    y2 = min(h, y + bh + pad_y)
                    proposed_boxes.append([x1, y1, x2, y2])

        # Non-Maximum Suppression / De-duplication of overlapping region proposals
        return self._non_max_suppression_boxes(proposed_boxes, overlap_thresh=0.45)

    def _non_max_suppression_boxes(self, boxes: List[List[int]], overlap_thresh: float) -> List[List[int]]:
        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(areas)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(boxes[i])

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1 + 1)
            h_inter = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w_inter * h_inter
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]

        return keep[:40]

    def preprocess_crop_for_dates(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Applies multi-pass enhancements tailored for dot-matrix and thermal packaging inks.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Upscale small text crops by 2.5x
        scale = 2.5 if max(crop.shape[:2]) < 300 else 1.5
        resized = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 1. CLAHE Contrast
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(resized)

        # 2. Otsu threshold
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphological close to bridge dot-matrix print gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

        # 4. Inverted threshold for bright text on dark surfaces
        otsu_inv = cv2.bitwise_not(otsu)

        return [clahe, otsu, closed, otsu_inv]

    def clean_ocr_typos(self, text: str) -> str:
        """
        Fixes common character confusions in packaging stamp OCR.
        """
        # Keyword typo corrections
        corrected = text
        corrected = re.sub(r'\bE[E|B]ST\b', 'BEST', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b[B|E]EF\s*ORE\b', 'BEFORE', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bEXP[I|1|L]RY\b', 'EXPIRY', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b[U|V]SE\s*B[Y|V]\b', 'USE BY', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bMF[G|D|C]\b', 'MFG', corrected, flags=re.IGNORECASE)

        # Fix OCR number substitutions inside date-like structures (e.g., 2O25/O1/28 -> 2025/01/28)
        def fix_date_digits(m):
            s = m.group(0)
            s = s.replace('O', '0').replace('o', '0').replace('D', '0')
            s = s.replace('I', '1').replace('l', '1').replace('|', '1')
            s = s.replace('S', '5').replace('s', '5')
            s = s.replace('B', '8')
            s = s.replace('Z', '2').replace('z', '2')
            return s

        # Target patterns that look like dates with letters
        date_typo_pat = r'\b(?:20[0-9OIlSBZ]{2}|[0-9OIlSBZ]{2})[-/. ][0-9OIlSBZ]{1,2}[-/. ][0-9OIlSBZ]{2,4}\b'
        corrected = re.sub(date_typo_pat, fix_date_digits, corrected)

        return corrected

    def extract_text_and_boxes(self, image_input) -> Dict[str, Any]:
        """
        Full dynamic extraction pipeline combining global OCR and localized candidate region proposals.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input
            rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        elif isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
            if cv_img is None:
                raise ValueError(f"Failed to read image at {image_input}")
            rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        else:
            raise TypeError("Unsupported image input type")

        h, w = cv_img.shape[:2]
        extracted_tokens: List[Dict[str, Any]] = []
        full_text_lines: List[str] = []

        if not self.tesseract_available:
            return {
                "full_text": "",
                "lines": [],
                "tokens": [],
                "image_dimensions": {"width": w, "height": h}
            }

        # 1. Global Full-Image Pass (for large typography, brand, ingredient list)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        try:
            global_data = pytesseract.image_to_data(
                clahe,
                output_type=pytesseract.Output.DICT,
                config=r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
            )
            n_boxes = len(global_data['text'])
            for i in range(n_boxes):
                text = global_data['text'][i].strip()
                conf = float(global_data['conf'][i])
                if text and conf > 30:
                    cleaned_t = self.clean_ocr_typos(text)
                    x = int(global_data['left'][i])
                    y = int(global_data['top'][i])
                    bw = int(global_data['width'][i])
                    bh = int(global_data['height'][i])

                    extracted_tokens.append({
                        "text": cleaned_t,
                        "confidence": round(conf, 1),
                        "bbox": [x, y, x + bw, y + bh],
                        "source": "global"
                    })

            raw_text = pytesseract.image_to_string(clahe, config=r'--oem 3 --psm 6')
            for line in raw_text.splitlines():
                cl = self.clean_ocr_typos(line.strip())
                if cl:
                    full_text_lines.append(cl)
        except Exception as e:
            logger.error(f"Global OCR error: {e}")

        # 2. Regional Proposal Pass (for small, dot-matrix, or low-contrast date stamps)
        candidate_regions = self.generate_candidate_regions(cv_img)
        for box in candidate_regions:
            x1, y1, x2, y2 = box
            crop = cv_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_variants = self.preprocess_crop_for_dates(crop)
            for variant in crop_variants:
                try:
                    # Specialized OCR config with date characters
                    crop_text = pytesseract.image_to_string(
                        variant,
                        config=r'--oem 3 --psm 6'
                    ).strip()

                    cleaned_crop = self.clean_ocr_typos(crop_text)

                    # If crop contains date patterns or expiry keywords, record it
                    if cleaned_crop and (
                        re.search(r'\b\d{2,4}[-/. ]\d{1,2}[-/. ]\d{2,4}\b', cleaned_crop) or
                        re.search(r'\b(?:EXP|BEST|BEFORE|USE|MFG|PKD|LOT|BATCH)\b', cleaned_crop, re.IGNORECASE)
                    ):
                        extracted_tokens.append({
                            "text": cleaned_crop,
                            "confidence": 85.0,
                            "bbox": box,
                            "source": "region_proposal"
                        })
                        full_text_lines.append(cleaned_crop)
                        break  # Found date in this variant, continue to next region
                except Exception:
                    pass

        # De-duplicate lines and construct aggregated text
        unique_lines = []
        seen_lines = set()
        for l in full_text_lines:
            if l not in seen_lines:
                seen_lines.add(l)
                unique_lines.append(l)

        aggregated_text = "\n".join(unique_lines)

        return {
            "full_text": aggregated_text,
            "lines": unique_lines,
            "tokens": extracted_tokens,
            "image_dimensions": {"width": w, "height": h}
        }
