# VeriSight Dataset Documentation

VeriSight integrates with four datasets situated in `datasets/raw/`:

## 1. ExpDate Dataset (`datasets/raw/expdate`)
- **Products-Real**: 1,102 real product packaging images with 1,244 ground truth date bounding boxes and 895 due marker annotations (`cls: date`, `cls: due`).
- **Date-Real & Date-Synth**: Isolated date stamp crops and synthetic training patches.

## 2. IMD2020 Manipulation Dataset (`datasets/raw/imd2020`)
- 414 manipulation categories.
- Contains authentic camera originals alongside digitally spliced/manipulated images and pixel-accurate binary ground-truth masks (`*_mask.png`).
- Used for evaluating the Error Level Analysis (ELA) and noise floor inconsistency algorithms.

## 3. Food Packaging OCR (`datasets/raw/foodpackagingocr`)
- Split into detection (`det`) and recognition (`rec`) tasks across 8,736 Train, 1,092 Validation, and 1,092 Test sets.
- Provides polygon annotations for diverse food packaging text.

## 4. OpenFoodFacts (`datasets/raw/openfoodfacts`)
- Manifest and catalog of real commercial product packaging images for production verification.

---

## Dataset Processing Pipeline

Use the scripts in `scripts/`:
1. `python scripts/inspect_dataset.py`: Scans and computes dataset statistics.
2. `python scripts/prepare_dataset.py`: Generates unified benchmark manifest in `datasets/processed/benchmark_manifest.json`.
3. `python scripts/split_dataset.py`: Splits benchmark manifest into Train (70%), Val (15%), and Test (15%) subsets.
