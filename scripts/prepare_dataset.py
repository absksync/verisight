import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"

def build_benchmark_manifest():
    """
    Builds a unified benchmark manifest aggregating real date samples and forgery samples
    for automated model evaluation.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 1. Index ExpDate samples
    expdate_ann = RAW_DIR / "expdate" / "products_real" / "Products-Real" / "train" / "annotations.json"
    expdate_imgs = RAW_DIR / "expdate" / "products_real" / "Products-Real" / "train" / "images"

    if expdate_ann.exists() and expdate_imgs.exists():
        with open(expdate_ann, "r") as f:
            data = json.load(f)

        for img_name, meta in list(data.items())[:100]:
            img_path = expdate_imgs / img_name
            if img_path.exists():
                dates = [a for a in meta.get("ann", []) if a.get("cls") == "date"]
                manifest.append({
                    "id": f"exp_{img_name}",
                    "path": str(img_path.relative_to(BASE_DIR)),
                    "type": "expdate_verification",
                    "ground_truth_date": dates[0].get("transcription") if dates else None,
                    "is_tampered": False
                })

    # 2. Index IMD2020 samples
    imd_root = RAW_DIR / "imd2020" / "IMD2020"
    if imd_root.exists():
        for d in list(imd_root.iterdir())[:50]:
            if d.is_dir():
                tampered = [f for f in d.glob("*.jpg") if not f.name.endswith("_orig.jpg")]
                origs = list(d.glob("*_orig.jpg"))
                if tampered:
                    manifest.append({
                        "id": f"imd_{tampered[0].stem}",
                        "path": str(tampered[0].relative_to(BASE_DIR)),
                        "type": "forgery_detection",
                        "is_tampered": True
                    })
                if origs:
                    manifest.append({
                        "id": f"imd_{origs[0].stem}",
                        "path": str(origs[0].relative_to(BASE_DIR)),
                        "type": "authentic_baseline",
                        "is_tampered": False
                    })

    out_file = PROCESSED_DIR / "benchmark_manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated unified benchmark manifest with {len(manifest)} items at {out_file}")

if __name__ == "__main__":
    build_benchmark_manifest()
