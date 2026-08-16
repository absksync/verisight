import pytest
import numpy as np
import cv2
from datetime import date
from backend.ml.date_extractor import DateExtractor
from backend.ml.tamper_detector import PackagingTamperDetector
from backend.ml.ocr_engine import OCREngine
from backend.services.verification_service import VerificationService
from backend.services.dataset_service import DatasetService

def test_date_extractor_valid():
    extractor = DateExtractor(reference_date=date(2026, 8, 14))
    ocr_result = {
        "full_text": "PROD 2026/01/01 \n EXP 2027/05/20 \n BATCH B99120",
        "lines": ["PROD 2026/01/01", "EXP 2027/05/20", "BATCH B99120"],
        "tokens": [
            {"text": "EXP", "bbox": [10, 10, 50, 30]},
            {"text": "2027/05/20", "bbox": [60, 10, 150, 30]}
        ]
    }
    res = extractor.extract(ocr_result)
    assert res["expiry_status"] == "VALID"
    assert res["expiry_date"] == "2027-05-20"
    assert res["mfg_date"] == "2026-01-01"
    assert res["days_remaining"] > 0
    assert res["batch_number"] == "B99120"

def test_date_extractor_expired():
    extractor = DateExtractor(reference_date=date(2026, 8, 14))
    ocr_result = {
        "full_text": "BEST BEFORE 15/04/2025",
        "lines": ["BEST BEFORE 15/04/2025"],
        "tokens": [{"text": "15/04/2025", "bbox": [10, 10, 100, 30]}]
    }
    res = extractor.extract(ocr_result)
    assert res["expiry_status"] == "EXPIRED"
    assert res["expiry_date"] == "2025-04-15"
    assert res["days_remaining"] < 0

def test_date_extractor_expiring_soon():
    extractor = DateExtractor(reference_date=date(2026, 8, 14))
    ocr_result = {
        "full_text": "USE BY 28/08/2026",
        "lines": ["USE BY 28/08/2026"],
        "tokens": [{"text": "28/08/2026", "bbox": [10, 10, 100, 30]}]
    }
    res = extractor.extract(ocr_result)
    assert res["expiry_status"] == "EXPIRING_SOON"
    assert res["days_remaining"] == 14

def test_tamper_detector_synthetic_image():
    detector = PackagingTamperDetector()
    # Create test packaging image
    img = np.full((300, 400, 3), 220, dtype=np.uint8)
    cv2.putText(img, "ORGANIC WHOLE MILK", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "EXP: 2027/12/31", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    report = detector.analyze(img)
    assert "tamper_score" in report
    assert "authenticity_score" in report
    assert "heatmap_image" in report
    assert report["heatmap_image"].startswith("data:image/jpeg;base64,")

def test_verification_service_end_to_end():
    service = VerificationService()
    # Generate image buffer
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    cv2.putText(img, "EXP 2027/10/15", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, buf = cv2.imencode(".jpg", img)
    img_bytes = buf.tobytes()

    result = service.verify_image(img_bytes, filename="test_carton.jpg")
    assert result["verdict"]["status"] in ["PASS", "WARNING", "REJECT"]
    assert "images" in result
    assert "annotated" in result["images"]
    assert "heatmap" in result["images"]

def test_dataset_service_catalog():
    ds = DatasetService()
    catalog = ds.get_sample_catalog(limit_per_dataset=5)
    assert len(catalog) > 0
    assert any(c["dataset"] == "ExpDate-Real" for c in catalog)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
