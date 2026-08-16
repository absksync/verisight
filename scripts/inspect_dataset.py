import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "raw"

def inspect_expdate():
    print("=" * 60)
    print("1. INSPECTING EXPDATE DATASET")
    print("=" * 60)
    ann_path = DATASETS_DIR / "expdate" / "products_real" / "Products-Real" / "train" / "annotations.json"
    if not ann_path.exists():
        print(f"File not found: {ann_path}")
        return

    with open(ann_path, "r") as f:
        data = json.load(f)

    print(f"Products-Real Train Images: {len(data)}")
    date_count = sum(1 for _, info in data.items() for a in info.get("ann", []) if a.get("cls") == "date")
    due_count = sum(1 for _, info in data.items() for a in info.get("ann", []) if a.get("cls") == "due")
    print(f"  • Date Bounding Boxes: {date_count}")
    print(f"  • Due Marker Bounding Boxes: {due_count}")

def inspect_imd2020():
    print("\n" + "=" * 60)
    print("2. INSPECTING IMD2020 FORGERY DATASET")
    print("=" * 60)
    imd_root = DATASETS_DIR / "imd2020" / "IMD2020"
    if not imd_root.exists():
        print(f"Directory not found: {imd_root}")
        return

    subdirs = [d for d in imd_root.iterdir() if d.is_dir()]
    print(f"Total Manipulation Folders: {len(subdirs)}")
    orig_count = sum(1 for d in subdirs if list(d.glob("*_orig.jpg")))
    mask_count = sum(len(list(d.glob("*_mask.png"))) for d in subdirs)
    tampered_count = sum(len([f for f in d.glob("*.jpg") if not f.name.endswith("_orig.jpg")]) for d in subdirs)

    print(f"  • Authentic Original Images: {orig_count}")
    print(f"  • Manipulated/Tampered Images: {tampered_count}")
    print(f"  • Ground Truth Binary Masks: {mask_count}")

def inspect_foodpackaging():
    print("\n" + "=" * 60)
    print("3. INSPECTING FOOD PACKAGING OCR DATASET")
    print("=" * 60)
    fp_root = DATASETS_DIR / "foodpackagingocr" / "FoodPackagingOCR" / "Food Packaging OCR Dataset"
    if not fp_root.exists():
        print(f"Directory not found: {fp_root}")
        return

    for task in ["det", "rec"]:
        task_dir = fp_root / task
        if task_dir.exists():
            for subset in ["train", "valid", "test"]:
                subset_dir = task_dir / subset
                labels_file = subset_dir / "labels.txt"
                count = 0
                if labels_file.exists():
                    with open(labels_file, "r") as f:
                        count = len(f.readlines())
                print(f"  • Task [{task.upper()}] - {subset.capitalize()}: {count} annotations")

if __name__ == "__main__":
    inspect_expdate()
    inspect_imd2020()
    inspect_foodpackaging()
    print("\nDataset inspection complete.")