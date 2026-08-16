# VeriSight: AI Packaging Integrity & Expiry Verification Engine

**VeriSight** is an AI-powered verification and forensic inspection system for packaged commercial goods, pharmaceuticals, and food products. It combines multi-layer **OCR date detection & shelf-life calculation** with **forensic image manipulation detection** to prevent counterfeit distribution, date tampering, and consumer safety hazards.

---

## Key Features

1. **Intelligent Expiry & Manufacturing Date OCR**:
   - Multi-format regex and contextual keyword recognition (`EXP`, `BEST BEFORE`, `USE BY`, `MFG`, `PKD`, `BATCH/LOT`).
   - Standardized ISO date parsing and countdown calculation (`days_remaining`, % shelf-life remaining).
   - Automated status classification: **`VALID`**, **`EXPIRING_SOON`**, or **`EXPIRED`**.

2. **Packaging Tampering & Forensic Forgery Detection**:
   - **Error Level Analysis (ELA)**: Re-compression differential analysis to pinpoint digitally altered or spliced date stamps.
   - **Sensor Noise Floor Inconsistency**: Local Laplacian standard deviation variance mapping.
   - **Edge Discontinuity & Gradient Anomalies**: Sobel gradient continuity analysis around typography.
   - **Thermal Forensic Heatmap**: Real-time visual overlay revealing modified hotspots and tampering risk percentage.

3. **Multi-Dataset Benchmarking & Catalog**:
   - Integrated with real product benchmarks: **ExpDate-Real**, **IMD2020 Manipulation Dataset**, **FoodPackagingOCR**, and **OpenFoodFacts**.
   - 1-click sample inspection and automated validation splits.

4. **High-Performance Full-Stack Application**:
   - **Backend**: FastAPI with OpenCV, Tesseract OCR, Pillow, and Pydantic.
   - **Frontend**: Modern React + Vite interactive dashboard with live webcam scanner, multi-layer inspection viewport (Visual Bounding Boxes / Forensic Heatmap / Raw), telemetry cards, and audit log history.

---

## Architecture Overview

```
verisight/
├── backend/
│   ├── api/
│   │   └── routes.py              # REST API Endpoints (/verify, /samples, /history, /metrics)
│   ├── ml/
│   │   ├── date_extractor.py      # Multi-pattern date parsing & shelf-life engine
│   │   ├── ocr_engine.py          # Adaptive image pre-processing & Tesseract OCR
│   │   └── tamper_detector.py     # Forensic ELA, noise inconsistency & thermal heatmap
│   ├── services/
│   │   ├── verification_service.py # Unified inspection verdict orchestrator
│   │   └── dataset_service.py     # Sample catalog loader for benchmark datasets
│   ├── utils/
│   │   └── image_utils.py         # Visual bounding box rendering & Base64 encoders
│   ├── app.py                     # FastAPI application entrypoint
│   └── config.py                  # Project paths and threshold configurations
├── frontend/                      # Modern React + Vite interactive UI
│   ├── src/
│   │   ├── App.jsx                # Main dashboard component
│   │   ├── index.css              # Custom design system
│   │   └── main.jsx
│   └── vite.config.js
├── datasets/                      # Raw and processed benchmark datasets
│   ├── raw/ (expdate, imd2020, foodpackagingocr, openfoodfacts)
│   └── processed/
├── scripts/
│   ├── inspect_dataset.py         # Dataset inspection and statistics
│   ├── prepare_dataset.py         # Unified benchmark manifest builder
│   └── split_dataset.py           # Train/Val/Test partitioning
├── tests/
│   └── test_pipeline.py           # Unit and integration test suite
└── requirements.txt
```

---

## Getting Started

### 1. Environment Setup

```bash
# Clone or navigate to the repository
cd verisight

# Activate Python Virtual Environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Running the Backend Server

```bash
# Start FastAPI backend on http://localhost:8000
python -m backend.app
```

### 3. Running the Frontend (Development Mode)

```bash
cd frontend
npm install
npm run dev
```

*The frontend runs on `http://localhost:5173` and proxies API calls to `http://localhost:8000`.*

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Dataset Tools

```bash
# Inspect all raw datasets
python scripts/inspect_dataset.py

# Build benchmark manifest
python scripts/prepare_dataset.py

# Generate Train/Val/Test splits
python scripts/split_dataset.py
```
