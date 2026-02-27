## ML Fraud Detection Pipeline Overview

This folder implements the **context-aware fraud detection** pipeline for product images with expiry dates.

### 1. User Input

- **Inputs**: product image, claimed delivery date (string).
- Entry point for batch processing is `ml/run_dataset.py`, which reads metadata from `ml/data/Hack_data/metadata.csv`.

### 2. OCR-Based Evidence Understanding

- `modules/ocr_module.py` uses **EasyOCR** to read text from the product image.
- It searches for expiry-like patterns (MM/YY, MM/YYYY, DD/MM/YYYY, YYYY, `EXP: 05/26`, etc.).
- The best match returns:
  - `expiry_text` string
  - bounding box of the region
  - OCR confidence score.

- `modules/expiry_logic.py` then:
  - Parses the `expiry_text` in multiple date formats.
  - Parses the provided delivery date (also tolerant to different formats).
  - Computes a **timeline risk** between 0 and 1:
    - `1.0` if the product **expired before delivery** (highly suspicious).
    - `0.0` if the expiry is **after or equal to** the delivery date.
    - Intermediate values (e.g., `0.3`, `0.5`) when OCR or dates are unreliable.

### 3. Image Forensic Analysis

- `modules/ela_module.py`
  - Runs **Error Level Analysis (ELA)** by recompressing the image to JPEG and taking the difference.
  - Produces:
    - a scalar score in \[0, 1\] (higher ⇒ more compression / edit artefacts),
    - an ELA heatmap saved under `ml/outputs/heatmaps/`.

- `modules/fft_module.py`
  - Applies a **2D FFT** on the grayscale image.
  - Uses the average magnitude in the frequency domain as a score in \[0, 1\] (higher ⇒ stronger high‑frequency / texture anomalies).

- `modules/vit_module.py`
  - Uses a pretrained **Vision Transformer** (`google/vit-base-patch16-224`) from Hugging Face.
  - Computes a risk score as `1 - top_class_probability`, so low model confidence leads to a higher risk value.

- `utils/image_utils.py`
  - Uses the OCR bounding box to draw a rectangle on the original image.
  - Saves a highlighted copy under `ml/outputs/highlighted/` for visual review.

### 4. Risk Fusion Engine

- `fusion.py`
  - Combines the per-module risk scores into a single **0–100 risk score**:
    - 40% ELA
    - 30% FFT
    - 20% ViT
    - 10% expiry‑timeline risk.

- `tags_engine.py`
  - Generates human-readable **tags** such as:
    - `"Compression Artifact"`
    - `"Texture Irregularity"`
    - `"Synthetic Pattern"`
    - `"OCR Confidence Drop"`.

- `confidence_engine.py`
  - Computes a simple consensus / agreement score from the module risks.

- `pipeline.py`
  - Orchestrates the whole flow for a single image:
    1. OCR expiry extraction.
    2. Context-aware expiry vs delivery check.
    3. ELA, FFT, ViT risk scoring.
    4. Risk fusion + decision thresholds:
       - **0–35** → Approve
       - **35–55** → Manual Review
       - **55+** → High Risk / Reject
    5. Generates tags and highlighted image path.

### 5. Batch Inference

- `run_dataset.py`:
  - Loads metadata from `ml/data/Hack_data/metadata.csv`.
  - Resolves image paths via `utils/metadata_loader.py`.
  - Runs `pipeline.run_pipeline` for each row.
  - Writes consolidated results to `ml/outputs/results.csv` with:
    - filename, category
    - final risk score and decision
    - timeline message and extracted expiry text
    - per-module scores (ELA, FFT, ViT, expiry risk)
    - OCR confidence and fusion confidence
    - tags and highlighted image path.

### Data Augmentation

No on-the-fly **data augmentation** is performed here because this pipeline is based on
pretrained / analytic models (OCR, ViT, ELA, FFT) and runs in **inference mode only**.
If you later add a supervised classifier on top of these features, augmentation can be
introduced in that separate training code.

