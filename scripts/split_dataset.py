import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
DATASETS_DIR = BASE_DIR / "datasets"

def split_processed_benchmark(train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    manifest_file = PROCESSED_DIR / "benchmark_manifest.json"
    if not manifest_file.exists():
        print("Run prepare_dataset.py first.")
        return

    with open(manifest_file, "r") as f:
        items = json.load(f)

    random.seed(seed)
    random.shuffle(items)

    n_total = len(items)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = items[:n_train]
    val_set = items[n_train:n_train + n_val]
    test_set = items[n_train + n_val:]

    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    for split_name, split_data in splits.items():
        out_path = PROCESSED_DIR / f"{split_name}_split.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Created {split_name} split with {len(split_data)} samples at {out_path}")

if __name__ == "__main__":
    split_processed_benchmark()
