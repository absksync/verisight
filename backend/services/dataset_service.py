import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from backend.config import (
    EXPDATE_DIR,
    FOODPACKAGING_DIR,
    IMD2020_DIR,
    OPENFOODFACTS_DIR
)
from backend.utils.image_utils import cv2_to_base64
import cv2

logger = logging.getLogger(__name__)

class DatasetService:
    """
    Manages and serves sample packaging and forensic datasets for interactive testing & benchmarking.
    """

    def __init__(self):
        self._cached_samples: Optional[List[Dict[str, Any]]] = None

    def get_sample_catalog(self, limit_per_dataset: int = 15) -> List[Dict[str, Any]]:
        """
        Gathers representative samples from:
        1. Products-Real (Expiry Date ground truth)
        2. FoodPackagingOCR (Packaging text detection/recognition)
        3. IMD2020 (Image manipulation & forgery detection)
        4. OpenFoodFacts (Real commercial packaging)
        """
        if self._cached_samples is not None:
            return self._cached_samples

        samples: List[Dict[str, Any]] = []

        # 1. ExpDate - Products Real
        expdate_ann_path = EXPDATE_DIR / "products_real" / "Products-Real" / "train" / "annotations.json"
        expdate_img_dir = EXPDATE_DIR / "products_real" / "Products-Real" / "train" / "images"

        if expdate_ann_path.exists() and expdate_img_dir.exists():
            try:
                with open(expdate_ann_path, "r") as f:
                    ann_data = json.load(f)

                count = 0
                for img_name, img_meta in ann_data.items():
                    img_path = expdate_img_dir / img_name
                    if img_path.exists():
                        dates = [a for a in img_meta.get("ann", []) if a.get("cls") == "date"]
                        dues = [a for a in img_meta.get("ann", []) if a.get("cls") == "due"]

                        date_text = dates[0].get("transcription", "N/A") if dates else "N/A"

                        samples.append({
                            "id": f"expdate_{img_name}",
                            "name": f"Product Expiry Sample ({img_name})",
                            "dataset": "ExpDate-Real",
                            "category": "Expiry & Due Verification",
                            "file_path": str(img_path),
                            "ground_truth": {
                                "date_text": date_text,
                                "date_regions": len(dates),
                                "due_markers": len(dues),
                                "is_tampered": False
                            },
                            "tags": ["Real Packaging", "Expiry Date", "Due Marker"]
                        })
                        count += 1
                        if count >= limit_per_dataset:
                            break
            except Exception as e:
                logger.error(f"Failed loading ExpDate dataset catalog: {e}")

        # 2. IMD2020 - Image Manipulation & Forgery
        imd_root = IMD2020_DIR / "IMD2020"
        if imd_root.exists():
            try:
                imd_dirs = [d for d in imd_root.iterdir() if d.is_dir()][:limit_per_dataset]
                for d in imd_dirs:
                    orig_files = list(d.glob("*_orig.jpg"))
                    tampered_files = [f for f in d.glob("*.jpg") if not f.name.endswith("_orig.jpg")]
                    mask_files = list(d.glob("*_mask.png"))

                    if tampered_files:
                        t_img = tampered_files[0]
                        samples.append({
                            "id": f"imd_{t_img.stem}",
                            "name": f"Tampered Packaging Sample ({d.name})",
                            "dataset": "IMD2020",
                            "category": "Manipulation & Forgery",
                            "file_path": str(t_img),
                            "ground_truth": {
                                "is_tampered": True,
                                "has_mask": len(mask_files) > 0
                            },
                            "tags": ["Tampered/Manipulated", "Forensic Test", "Altered Substrate"]
                        })

                    if orig_files and len(samples) < limit_per_dataset * 2:
                        o_img = orig_files[0]
                        samples.append({
                            "id": f"imd_orig_{o_img.stem}",
                            "name": f"Authentic Reference Sample ({d.name})",
                            "dataset": "IMD2020",
                            "category": "Authentic Substrate",
                            "file_path": str(o_img),
                            "ground_truth": {
                                "is_tampered": False
                            },
                            "tags": ["Authentic", "Baseline"]
                        })
            except Exception as e:
                logger.error(f"Failed loading IMD2020 catalog: {e}")

        # 3. Food Packaging OCR
        fp_dir = FOODPACKAGING_DIR / "FoodPackagingOCR" / "Food Packaging OCR Dataset" / "det" / "valid"
        fp_labels = fp_dir / "labels.txt"
        fp_imgs = fp_dir / "images"

        if fp_labels.exists() and fp_imgs.exists():
            try:
                with open(fp_labels, "r") as f:
                    fp_lines = f.readlines()[:limit_per_dataset]

                for line in fp_lines:
                    parts = line.strip().split("\t")
                    if parts:
                        img_file = parts[0]
                        img_path = fp_imgs / img_file
                        if img_path.exists():
                            samples.append({
                                "id": f"fp_{img_file}",
                                "name": f"Food OCR Sample ({img_file})",
                                "dataset": "FoodPackagingOCR",
                                "category": "Packaging Text Recognition",
                                "file_path": str(img_path),
                                "ground_truth": {
                                    "is_tampered": False,
                                    "ocr_sample": True
                                },
                                "tags": ["Text Detection", "Food Packaging", "OCR"]
                            })
            except Exception as e:
                logger.error(f"Failed loading FoodPackagingOCR catalog: {e}")

        self._cached_samples = samples
        return samples

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        catalog = self.get_sample_catalog(limit_per_dataset=50)
        for item in catalog:
            if item["id"] == sample_id:
                return item
        return None

    def get_sample_image_base64(self, sample_id: str) -> Optional[str]:
        item = self.get_sample_by_id(sample_id)
        if not item or not os.path.exists(item["file_path"]):
            return None

        cv_img = cv2.imread(item["file_path"])
        if cv_img is None:
            return None
        return cv2_to_base64(cv_img, quality=80)
