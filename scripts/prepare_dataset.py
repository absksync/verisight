import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"


def build_benchmark_manifest():
    """
    Builds a unified benchmark manifest aggregating:
    1. ExpDate-Real: real packaging images with labelled expiry date regions
    2. IMD2020: image manipulation / forgery pairs (tampered + authentic)
    3. OpenFoodFacts: real commercial packaging photos

    Output: datasets/processed/benchmark_manifest.json
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # ── 1. ExpDate-Real ──────────────────────────────────────────────────────
    expdate_ann = (
        RAW_DIR / "expdate" / "products_real" / "Products-Real"
        / "train" / "annotations.json"
    )
    expdate_imgs = (
        RAW_DIR / "expdate" / "products_real" / "Products-Real"
        / "train" / "images"
    )

    if expdate_ann.exists() and expdate_imgs.exists():
        with open(expdate_ann, "r") as f:
            data = json.load(f)

        count = 0
        for img_name, meta in data.items():
            img_path = expdate_imgs / img_name
            if not img_path.exists():
                continue
            dates = [a for a in meta.get("ann", []) if a.get("cls") == "date"]
            dues  = [a for a in meta.get("ann", []) if a.get("cls") == "due"]
            manifest.append({
                "id": f"exp_{img_name}",
                "path": str(img_path.relative_to(BASE_DIR)),
                "type": "expdate_verification",
                "dataset": "ExpDate-Real",
                "ground_truth_date": dates[0].get("transcription") if dates else None,
                "ground_truth_due":  dues[0].get("transcription")  if dues  else None,
                "is_tampered": False,
                "has_date_annotation": len(dates) > 0,
            })
            count += 1

        logger.info(f"[ExpDate-Real] Indexed {count} annotated images")
    else:
        logger.warning(f"[ExpDate-Real] Not found at {expdate_ann}")

    # ── 2. IMD2020 — Tamper Detection ─────────────────────────────────────────
    imd_root = RAW_DIR / "imd2020" / "IMD2020"
    tampered_count = 0
    authentic_count = 0

    if imd_root.exists():
        for sample_dir in imd_root.iterdir():
            if not sample_dir.is_dir():
                continue

            # Authentic originals
            for orig in sample_dir.glob("*_orig.jpg"):
                manifest.append({
                    "id": f"imd_auth_{orig.stem}",
                    "path": str(orig.relative_to(BASE_DIR)),
                    "type": "tamper_detection",
                    "dataset": "IMD2020",
                    "is_tampered": False,
                    "has_date_annotation": False,
                })
                authentic_count += 1

            # Tampered images (not ending in _orig.jpg, not masks)
            for tampered in sample_dir.glob("*.jpg"):
                if tampered.name.endswith("_orig.jpg"):
                    continue
                mask_path = tampered.with_name(tampered.stem + "_mask.png")
                manifest.append({
                    "id": f"imd_tamp_{tampered.stem}",
                    "path": str(tampered.relative_to(BASE_DIR)),
                    "type": "tamper_detection",
                    "dataset": "IMD2020",
                    "is_tampered": True,
                    "mask_path": str(mask_path.relative_to(BASE_DIR)) if mask_path.exists() else None,
                    "has_date_annotation": False,
                })
                tampered_count += 1

        logger.info(
            f"[IMD2020] Indexed {authentic_count} authentic + {tampered_count} tampered images"
        )
    else:
        logger.warning(f"[IMD2020] Not found at {imd_root}")

    # ── 3. OpenFoodFacts ──────────────────────────────────────────────────────
    off_root = RAW_DIR / "openfoodfacts" / "verisight_openfoodfacts" / "data"
    off_count = 0

    if off_root.exists():
        for img_path in off_root.rglob("*.jpg"):
            manifest.append({
                "id": f"off_{img_path.stem}",
                "path": str(img_path.relative_to(BASE_DIR)),
                "type": "packaging_sample",
                "dataset": "OpenFoodFacts",
                "is_tampered": False,
                "has_date_annotation": False,
            })
            off_count += 1

        logger.info(f"[OpenFoodFacts] Indexed {off_count} commercial packaging images")
    else:
        logger.warning(f"[OpenFoodFacts] Not found at {off_root}")

    # ── Save manifest ─────────────────────────────────────────────────────────
    out_file = PROCESSED_DIR / "benchmark_manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"✅ Manifest saved: {len(manifest)} total items → {out_file}"
    )
    return manifest


if __name__ == "__main__":
    build_benchmark_manifest()
