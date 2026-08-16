from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.services.verification_service import VerificationService
from backend.services.dataset_service import DatasetService
import base64
import os

router = APIRouter(prefix="/api")

# Service Singletons
verification_service = VerificationService()
dataset_service = DatasetService()

class Base64VerifyRequest(BaseModel):
    image_base64: str
    filename: Optional[str] = "image.jpg"
    product_name: Optional[str] = None

class SampleVerifyRequest(BaseModel):
    sample_id: str

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "VeriSight Engine", "tesseract": verification_service.ocr_engine.tesseract_available}

@router.post("/verify")
async def verify_image_upload(
    file: UploadFile = File(...),
    product_name: Optional[str] = Form(None)
):
    """
    Primary endpoint: inspect an uploaded packaging image for expiry dates,
    text markers, and forensic tampering.
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = verification_service.verify_image(
            image_bytes=contents,
            filename=file.filename or "upload.jpg",
            product_name_hint=product_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-base64")
async def verify_image_base64(req: Base64VerifyRequest):
    """
    Inspects image passed as base64 string (useful for live webcam feeds).
    """
    try:
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_str)

        result = verification_service.verify_image(
            image_bytes=image_bytes,
            filename=req.filename or "webcam_capture.jpg",
            product_name_hint=req.product_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/samples")
def list_dataset_samples(limit: int = Query(30, ge=1, le=100)):
    """
    Returns curated sample images from ExpDate-Real, IMD2020, FoodPackagingOCR, and OpenFoodFacts.
    """
    return dataset_service.get_sample_catalog(limit_per_dataset=limit)

@router.get("/samples/{sample_id}")
def get_sample_details(sample_id: str):
    sample = dataset_service.get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return sample

@router.post("/samples/verify")
def verify_sample(req: SampleVerifyRequest):
    """
    Directly runs verification on a sample from the pre-indexed dataset.
    """
    sample = dataset_service.get_sample_by_id(req.sample_id)
    if not sample or not os.path.exists(sample["file_path"]):
        raise HTTPException(status_code=404, detail="Sample image file not found.")

    try:
        with open(sample["file_path"], "rb") as f:
            image_bytes = f.read()

        result = verification_service.verify_image(
            image_bytes=image_bytes,
            filename=os.path.basename(sample["file_path"]),
            product_name_hint=sample["name"]
        )
        result["dataset_metadata"] = sample
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_inspection_history():
    """
    Returns recent inspection logs with filterable status.
    """
    return verification_service.get_history()

@router.get("/metrics")
def get_system_metrics():
    """
    Returns operational stats (pass/warning/reject counts, pass rate, avg speed).
    """
    return verification_service.get_metrics()
