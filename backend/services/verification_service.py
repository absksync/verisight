import cv2
import numpy as np
from PIL import Image
import io
import time
import uuid
from typing import Dict, Any, List, Optional
import logging
from backend.ml.ocr_engine import OCREngine
from backend.ml.date_extractor import DateExtractor
from backend.ml.tamper_detector import PackagingTamperDetector
from backend.utils.image_utils import draw_annotations, cv2_to_base64, pil_to_base64

logger = logging.getLogger(__name__)

class VerificationService:
    """
    Central VeriSight Engine.
    Coordinates OCR, date analysis, shelf life verification, and tamper forensics
    to produce a comprehensive inspection report.
    """

    def __init__(self):
        self.ocr_engine = OCREngine()
        self.date_extractor = DateExtractor()
        self.tamper_detector = PackagingTamperDetector()
        self.inspection_history: List[Dict[str, Any]] = []

    def verify_image(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        product_name_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs complete verification workflow on image bytes.
        """
        start_time = time.time()
        inspection_id = f"VS-{str(uuid.uuid4())[:8].upper()}"

        # Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Invalid or corrupted image format.")

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        h, w = cv_img.shape[:2]

        # 1. Run Tamper Detection
        tamper_report = self.tamper_detector.analyze(cv_img)

        # 2. Run OCR Extraction
        ocr_report = self.ocr_engine.extract_text_and_boxes(cv_img)

        # 3. Extract & Validate Dates
        date_report = self.date_extractor.extract(ocr_report)

        # 4. Synthesize Overall Verdict
        # Rules:
        # - If Tamper status == "TAMPERED" -> REJECT (Fraud/Tamper Risk)
        # - If Expiry status == "EXPIRED" -> REJECT (Expired product)
        # - If Expiry status == "EXPIRING_SOON" or Tamper status == "SUSPICIOUS" -> WARNING
        # - If Expiry status == "VALID" and Tamper status == "AUTHENTIC" -> PASS
        # - If No date detected -> WARNING (Unverified date)

        verdict_status = "PASS"
        verdict_color = "emerald"
        action_recommendation = "APPROVE FOR DISTRIBUTION / CONSUMPTION"
        reasons: List[str] = []

        is_tampered = tamper_report.get("status") == "TAMPERED"
        is_suspicious = tamper_report.get("status") == "SUSPICIOUS"
        is_expired = date_report.get("expiry_status") == "EXPIRED"
        is_expiring = date_report.get("expiry_status") == "EXPIRING_SOON"
        is_unknown_date = date_report.get("expiry_status") == "UNKNOWN"

        if is_tampered:
            verdict_status = "REJECT"
            verdict_color = "rose"
            action_recommendation = "REJECT & QUARANTINE: Digital or physical packaging manipulation detected."
            reasons.append(f"Forensic Tamper Score: {int(tamper_report['tamper_score']*100)}% (Hotspots detected on packaging)")

        if is_expired:
            verdict_status = "REJECT"
            verdict_color = "rose"
            action_recommendation = "REJECT: Product is past expiration date."
            reasons.append(f"Expired {abs(date_report['days_remaining'])} days ago on {date_report['expiry_date']}")

        if not is_tampered and not is_expired:
            if is_suspicious:
                verdict_status = "WARNING"
                verdict_color = "amber"
                action_recommendation = "MANUAL AUDIT: Irregular noise or compression detected."
                reasons.append(f"Packaging substrate has suspicious compression variance (Score: {int(tamper_report['tamper_score']*100)}%)")

            if is_expiring:
                verdict_status = "WARNING"
                verdict_color = "amber"
                action_recommendation = "EXPIRING SOON: Prioritize shelf clearance or discounting."
                reasons.append(f"Expires in {date_report['days_remaining']} days on {date_report['expiry_date']}")

            if is_unknown_date:
                verdict_status = "WARNING"
                verdict_color = "amber"
                action_recommendation = "UNVERIFIED: Expiry date not clearly detected in visible packaging area."
                reasons.append("No unambiguous EXP / BB date string found in scanned area.")

        if verdict_status == "PASS":
            reasons.append(f"Authentic packaging substrate ({tamper_report['authenticity_score']}% confidence)")
            reasons.append(f"Fresh & Valid. {date_report['days_remaining']} days remaining (Expires: {date_report['expiry_date']})")

        # 5. Render Visual Annotations
        annotated_cv = draw_annotations(cv_img, date_report, tamper_report, ocr_report.get("tokens", []))
        annotated_b64 = cv2_to_base64(annotated_cv, quality=85)
        raw_b64 = cv2_to_base64(cv_img, quality=80)

        elapsed = round(time.time() - start_time, 2)

        result = {
            "id": inspection_id,
            "filename": filename,
            "product_name": product_name_hint or "Packaged Commercial Good",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": elapsed,
            "verdict": {
                "status": verdict_status,
                "color": verdict_color,
                "action": action_recommendation,
                "reasons": reasons,
                "overall_score": round(
                    (tamper_report["authenticity_score"] * 0.5) +
                    (100.0 if date_report["expiry_status"] == "VALID" else (50.0 if date_report["expiry_status"] == "EXPIRING_SOON" else 0.0)) * 0.5,
                    1
                )
            },
            "expiry_analysis": date_report,
            "tamper_analysis": tamper_report,
            "ocr_analysis": {
                "text_summary": ocr_report.get("full_text", "")[:300],
                "total_tokens": len(ocr_report.get("tokens", [])),
                "tokens": ocr_report.get("tokens", [])[:50]
            },
            "images": {
                "raw": raw_b64,
                "annotated": annotated_b64,
                "heatmap": tamper_report.get("heatmap_image")
            }
        }

        # Keep history in memory (latest 50)
        self.inspection_history.insert(0, {
            "id": result["id"],
            "filename": result["filename"],
            "product_name": result["product_name"],
            "timestamp": result["timestamp"],
            "verdict_status": result["verdict"]["status"],
            "expiry_status": date_report["expiry_status"],
            "tamper_status": tamper_report["status"],
            "expiry_date": date_report["expiry_date"],
            "authenticity_score": tamper_report["authenticity_score"],
            "processing_time": elapsed
        })
        if len(self.inspection_history) > 50:
            self.inspection_history.pop()

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        return self.inspection_history

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self.inspection_history)
        if total == 0:
            return {
                "total_inspected": 0,
                "passed_count": 0,
                "warning_count": 0,
                "rejected_count": 0,
                "pass_rate_percentage": 100.0,
                "avg_processing_time_sec": 0.0
            }

        passed = sum(1 for h in self.inspection_history if h["verdict_status"] == "PASS")
        warnings = sum(1 for h in self.inspection_history if h["verdict_status"] == "WARNING")
        rejected = sum(1 for h in self.inspection_history if h["verdict_status"] == "REJECT")
        avg_time = sum(h["processing_time"] for h in self.inspection_history) / total

        return {
            "total_inspected": total,
            "passed_count": passed,
            "warning_count": warnings,
            "rejected_count": rejected,
            "pass_rate_percentage": round((passed / total) * 100.0, 1),
            "avg_processing_time_sec": round(avg_time, 2)
        }
