import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default reference date for verification — always today
DEFAULT_REFERENCE_DATE = date.today()

class DateExtractor:
    """
    Extracts, normalizes, and verifies expiration & manufacturing dates
    from packaging text and localized OCR tokens.
    """

    EXPIRY_KEYWORDS = [
        r'\bEXP(?:IRY)?(?:\s*DATE)?\b',
        r'\bBEST\s*BEFORE\b',
        r'\bUSE\s*BY\b',
        r'\bBB\b',
        r'\bBBD\b',
        r'\bCONSUME\s*(?:BEFORE|BY)\b',
        r'\bVALID\s*(?:THRU|UNTIL|TO)\b',
        r'\bDUE\b',
        r'\bVAL\b',
        r'\bE:\b'
    ]

    MFG_KEYWORDS = [
        r'\bMFG(?:\s*DATE)?\b',
        r'\bMFD\b',
        r'\bPROD(?:UCTION)?(?:\s*DATE)?\b',
        r'\bPRD\b',
        r'\bPKD\b',
        r'\bPACKED(?:\s*ON)?\b',
        r'\bDOM\b',
        r'\bM:\b',
        r'\bP:\b'
    ]

    BATCH_KEYWORDS = [
        r'\bBATCH(?:\s*NO)?\b',
        r'\bLOT(?:\s*NO)?\b',
        r'\bB\.?NO\b',
        r'\bL\.?NO\b',
        r'\bBN\b'
    ]

    # Full 3-part dates have highest priority, followed by alphanumeric, followed by 2-part month-year
    DATE_PATTERNS = [
        # YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        (r'(?<!\d)(20[1-3][0-9])[-/. ](0[1-9]|1[0-2])[-/. ](0[1-9]|[12][0-9]|3[01])(?!\d)', '%Y/%m/%d'),
        # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        (r'(?<!\d)(0[1-9]|[12][0-9]|3[01])[-/. ](0[1-9]|1[0-2])[-/. ](20[1-3][0-9])(?!\d)', '%d/%m/%Y'),
        # MM/DD/YYYY, MM-DD-YYYY
        (r'(?<!\d)(0[1-9]|1[0-2])[-/. ](0[1-9]|[12][0-9]|3[01])[-/. ](20[1-3][0-9])(?!\d)', '%m/%d/%Y'),
        # DD MMM YYYY / DD-MMM-YYYY / DD MMM YY (e.g. 14 AUG 2026, 28-JAN-25)
        (r'(?<!\d)(0[1-9]|[12][0-9]|3[01])[\s\-\/.](JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[\s\-\/.](20[1-3][0-9]|[1-3][0-9])(?!\d)', 'ALPHA_DMY'),
        # DD/MM/YY (e.g. 28/01/25)
        (r'(?<!\d)(0[1-9]|[12][0-9]|3[01])[-/. ](0[1-9]|1[0-2])[-/. ]([1-3][0-9])(?!\d)', '%d/%m/%y'),
        # MMM YYYY / MMM-YY (e.g. AUG 2026, OCT 25)
        (r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[\s\-\/.](20[1-3][0-9]|[1-3][0-9])(?!\d)', 'ALPHA_MY'),
        # MM/YYYY (e.g. 08/2026, 12/2027)
        (r'(?<!\d)(0[1-9]|1[0-2])[-/. ](20[1-3][0-9])(?!\d)', '%m/%Y'),
        # MM/YY (e.g. 08/26)
        (r'(?<!\d)(0[1-9]|1[0-2])[-/. ]([1-3][0-9])(?!\d)', '%m/%y'),
        # Compact 8-digit: YYYYMMDD (e.g., 20260814)
        (r'(?<!\d)(20[1-3][0-9])(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?!\d)', 'COMPACT_YMD'),
        # Compact 6-digit: DDMMYY
        (r'(?<!\d)(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[0-2])([1-3][0-9])(?!\d)', 'COMPACT_DMY')
    ]

    MONTH_MAP = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    def __init__(self, reference_date: Optional[date] = None):
        self.ref_date = reference_date or DEFAULT_REFERENCE_DATE

    def parse_date_string(self, date_str: str, pattern_type: str, match_groups: Tuple) -> Optional[date]:
        try:
            if pattern_type == '%Y/%m/%d':
                y, m, d = int(match_groups[0]), int(match_groups[1]), int(match_groups[2])
                return date(y, m, d)
            elif pattern_type == '%d/%m/%Y':
                d, m, y = int(match_groups[0]), int(match_groups[1]), int(match_groups[2])
                return date(y, m, d)
            elif pattern_type == '%m/%d/%Y':
                m, d, y = int(match_groups[0]), int(match_groups[1]), int(match_groups[2])
                return date(y, m, d)
            elif pattern_type == 'ALPHA_DMY':
                d = int(match_groups[0])
                m_str = match_groups[1].upper()[:3]
                m = self.MONTH_MAP.get(m_str, 1)
                y_raw = int(match_groups[2])
                y = y_raw if y_raw > 100 else 2000 + y_raw
                return date(y, m, d)
            elif pattern_type == 'ALPHA_MY':
                m_str = match_groups[0].upper()[:3]
                m = self.MONTH_MAP.get(m_str, 1)
                y_raw = int(match_groups[1])
                y = y_raw if y_raw > 100 else 2000 + y_raw
                d = 28 if m == 2 else (30 if m in [4, 6, 9, 11] else 31)
                return date(y, m, d)
            elif pattern_type == '%m/%Y':
                m = int(match_groups[0])
                y = int(match_groups[1])
                d = 28 if m == 2 else (30 if m in [4, 6, 9, 11] else 31)
                return date(y, m, d)
            elif pattern_type == '%m/%y':
                m = int(match_groups[0])
                y_raw = int(match_groups[1])
                y = 2000 + y_raw
                d = 28 if m == 2 else (30 if m in [4, 6, 9, 11] else 31)
                return date(y, m, d)
            elif pattern_type == '%d/%m/%y':
                d = int(match_groups[0])
                m = int(match_groups[1])
                y_raw = int(match_groups[2])
                y = 2000 + y_raw
                return date(y, m, d)
            elif pattern_type == 'COMPACT_YMD':
                y, m, d = int(match_groups[0]), int(match_groups[1]), int(match_groups[2])
                return date(y, m, d)
            elif pattern_type == 'COMPACT_DMY':
                d, m, y_raw = int(match_groups[0]), int(match_groups[1]), int(match_groups[2])
                y = 2000 + y_raw
                return date(y, m, d)
        except Exception as e:
            logger.debug(f"Date conversion error for {date_str}: {e}")
            return None
        return None

    def find_associated_bbox(self, text_snippet: str, tokens: List[Dict[str, Any]]) -> Optional[List[int]]:
        matched_boxes = []
        cleaned_snippet = re.sub(r'[\s\-/. :]', '', text_snippet.upper())

        for token in tokens:
            t_text = re.sub(r'[\s\-/. :]', '', token.get("text", "").upper())
            if t_text and (t_text in cleaned_snippet or cleaned_snippet in t_text or any(part in cleaned_snippet for part in t_text.split())):
                matched_boxes.append(token.get("bbox"))

        if matched_boxes:
            x1 = min([b[0] for b in matched_boxes])
            y1 = min([b[1] for b in matched_boxes])
            x2 = max([b[2] for b in matched_boxes])
            y2 = max([b[3] for b in matched_boxes])
            return [x1, y1, x2, y2]
        return None

    def extract(self, ocr_result: Dict[str, Any], custom_ref_date: Optional[date] = None) -> Dict[str, Any]:
        ref = custom_ref_date or self.ref_date
        full_text = ocr_result.get("full_text", "")
        tokens = ocr_result.get("tokens", [])
        lines = ocr_result.get("lines", [])

        extracted_dates: List[Dict[str, Any]] = []
        detected_batch: Optional[str] = None
        detected_batch_bbox: Optional[List[int]] = None

        # 1. Search for batch/lot numbers
        for batch_pat in self.BATCH_KEYWORDS:
            m = re.search(batch_pat + r'[\s:.\-#]*([A-Z0-9]{4,15})\b', full_text, re.IGNORECASE)
            if m:
                detected_batch = m.group(1).strip()
                detected_batch_bbox = self.find_associated_bbox(detected_batch, tokens)
                break

        # 2. Iterate through lines and text blocks
        all_text_blocks = lines if lines else [full_text]

        for block in all_text_blocks:
            matched_spans = []
            for pattern, ptype in self.DATE_PATTERNS:
                for match in re.finditer(pattern, block, re.IGNORECASE):
                    span = match.span()
                    if any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                        continue

                    matched_str = match.group(0)
                    parsed_dt = self.parse_date_string(matched_str, ptype, match.groups())
                    if not parsed_dt:
                        continue

                    matched_spans.append(span)

                    start_idx = max(0, match.start() - 30)
                    end_idx = min(len(block), match.end() + 30)
                    context_window = block[start_idx:end_idx].upper()

                    is_mfg = any(re.search(kw, context_window, re.IGNORECASE) for kw in self.MFG_KEYWORDS)
                    is_exp = any(re.search(kw, context_window, re.IGNORECASE) for kw in self.EXPIRY_KEYWORDS)

                    if is_exp and not is_mfg:
                        date_type = "EXPIRY"
                    elif is_mfg and not is_exp:
                        date_type = "MANUFACTURE"
                    else:
                        # If unlabelled, default to EXPIRY (as commercial date stamps on food/pharma indicate best before)
                        date_type = "EXPIRY"

                    bbox = self.find_associated_bbox(matched_str, tokens)

                    extracted_dates.append({
                        "raw_text": matched_str,
                        "formatted_date": parsed_dt.strftime("%Y-%m-%d"),
                        "date_obj": parsed_dt,
                        "type": date_type,
                        "context": context_window.strip(),
                        "bbox": bbox
                    })

        # De-duplicate
        unique_dates = []
        seen = set()
        for d in extracted_dates:
            key = (d["formatted_date"], d["type"])
            if key not in seen:
                seen.add(key)
                unique_dates.append(d)

        # Disambiguate if multiple dates found without explicit tags
        if len(unique_dates) == 2 and all(d["type"] == "EXPIRY" for d in unique_dates):
            # Sort chronologically: earlier date is MFG, later date is EXPIRY
            unique_dates.sort(key=lambda x: x["date_obj"])
            unique_dates[0]["type"] = "MANUFACTURE"
            unique_dates[1]["type"] = "EXPIRY"

        expiry_candidates = [d for d in unique_dates if d["type"] == "EXPIRY"]
        mfg_candidates = [d for d in unique_dates if d["type"] == "MANUFACTURE"]

        # Sort expiry candidates: prefer furthest date
        expiry_candidates.sort(key=lambda x: x["date_obj"], reverse=True)
        mfg_candidates.sort(key=lambda x: x["date_obj"])

        primary_expiry = expiry_candidates[0] if expiry_candidates else None
        primary_mfg = mfg_candidates[0] if mfg_candidates else None

        expiry_status = "UNKNOWN"
        days_remaining = None
        shelf_life_percentage = None
        status_message = "No expiration date detected."

        if primary_expiry:
            exp_date = primary_expiry["date_obj"]
            delta = (exp_date - ref).days
            days_remaining = delta

            if delta < 0:
                expiry_status = "EXPIRED"
                status_message = f"Product EXPIRED {abs(delta)} days ago (Expiry: {exp_date.strftime('%d %b %Y')}). Do not consume or sell."
            elif delta <= 30:
                expiry_status = "EXPIRING_SOON"
                status_message = f"Product is EXPIRING SOON in {delta} days (Expiry: {exp_date.strftime('%d %b %Y')})."
            else:
                expiry_status = "VALID"
                status_message = f"Product is FRESH & VALID. {delta} days remaining (Expiry: {exp_date.strftime('%d %b %Y')})."

            if primary_mfg:
                mfg_date = primary_mfg["date_obj"]
                total_lifespan = (exp_date - mfg_date).days
                if total_lifespan > 0:
                    elapsed = (ref - mfg_date).days
                    remaining_pct = max(0.0, min(100.0, ((total_lifespan - elapsed) / total_lifespan) * 100.0))
                    shelf_life_percentage = round(remaining_pct, 1)

        return {
            "expiry_status": expiry_status,
            "status_message": status_message,
            "days_remaining": days_remaining,
            "shelf_life_percentage": shelf_life_percentage,
            "reference_date": ref.strftime("%Y-%m-%d"),
            "expiry_date": primary_expiry["formatted_date"] if primary_expiry else None,
            "expiry_raw": primary_expiry["raw_text"] if primary_expiry else None,
            "expiry_bbox": primary_expiry["bbox"] if primary_expiry else None,
            "mfg_date": primary_mfg["formatted_date"] if primary_mfg else None,
            "mfg_raw": primary_mfg["raw_text"] if primary_mfg else None,
            "mfg_bbox": primary_mfg["bbox"] if primary_mfg else None,
            "batch_number": detected_batch,
            "batch_bbox": detected_batch_bbox,
            "all_extracted_dates": [
                {
                    "raw": d["raw_text"],
                    "formatted": d["formatted_date"],
                    "type": d["type"],
                    "bbox": d["bbox"]
                }
                for d in unique_dates
            ]
        }
