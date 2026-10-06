# VeriSight

AI packaging integrity and expiry verification for retail, food, and pharmaceutical products.

<p>
  <a href="#launch-pad"><img alt="Launch pad" src="https://img.shields.io/badge/open-launch%20pad-111827?style=for-the-badge"></a>
  <a href="#api-playground"><img alt="API playground" src="https://img.shields.io/badge/try-api%20playground-2563eb?style=for-the-badge"></a>
  <a href="#inspection-report"><img alt="Inspection report" src="https://img.shields.io/badge/read-inspection%20report-16a34a?style=for-the-badge"></a>
  <a href="#architecture"><img alt="Architecture" src="https://img.shields.io/badge/trace-system%20flow-7c3aed?style=for-the-badge"></a>
</p>

VeriSight turns a package photo into a decision-ready inspection: OCR finds expiry evidence, date logic checks shelf-life, forensic analysis searches for manipulation, and the dashboard renders annotated images, heatmaps, metrics, and exportable reports.

```text
Scan package -> Read dates -> Detect tampering -> Explain verdict -> Export audit
```

## Launch Pad

Choose the route that matches what you want to do right now.

| I want to... | Open this | Outcome |
|---|---|---|
| See it running fast | [Quick Start](#quick-start) | Local backend and React dashboard |
| Ship it in one command | [Docker](#docker) | Full app on `localhost:8000` |
| Test the engine directly | [API Playground](#api-playground) | Copyable API calls |
| Decode a verdict | [Inspection Report](#inspection-report) | What each result field means |
| Understand the internals | [Architecture](#architecture) | System flow, modules, and decision logic |
| Prepare benchmarks | [Datasets](#datasets) | Dataset manifests and splits |

<details open>
<summary><strong>60-second demo path</strong></summary>

```bash
python -m backend.app
```

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`, drop in a package image, switch between `annotated`, `heatmap`, and `raw`, and export the JSON report.

</details>

<details>
<summary><strong>What makes this different from plain OCR?</strong></summary>

VeriSight does not stop after reading text. It cross-checks extracted dates with shelf-life rules, inspects the image substrate for manipulation signals, and returns a verdict with reasons. That means a clear future expiry can still be rejected if the date region looks spliced, repainted, or digitally altered.

</details>

## Product Tour

<details open>
<summary><strong>Interactive inspection workflow</strong></summary>

```mermaid
flowchart LR
    A[Choose input] --> B{Upload, webcam, or sample?}
    B -->|Upload| C[POST /api/verify]
    B -->|Webcam| D[POST /api/verify-base64]
    B -->|Sample| E[POST /api/samples/verify]
    C --> F[OCR tokens]
    D --> F
    E --> F
    C --> G[Forensic maps]
    D --> G
    E --> G
    F --> H[Date intelligence]
    G --> I[Tamper score]
    H --> J{Verdict engine}
    I --> J
    J -->|PASS| K[Approve]
    J -->|WARNING| L[Manual review]
    J -->|REJECT| M[Quarantine]
```

</details>

<details>
<summary><strong>Dashboard controls</strong></summary>

| Control | What it does |
|---|---|
| Upload panel | Drag in a packaging image and run a full inspection |
| Camera mode | Capture a live frame and send it to `/api/verify-base64` |
| Sample browser | Pull curated benchmark samples into the scanner |
| View mode switcher | Compare annotated, heatmap, and raw image layers |
| History rail | Review recent verdicts and processing times |
| Export button | Download the full inspection JSON |

</details>

<details>
<summary><strong>Core AI signals</strong></summary>

- OCR extraction with Tesseract and adaptive OpenCV preprocessing.
- Date parsing for expiry, best-before, manufacturing, packing, batch, and lot markers.
- Shelf-life status classification: `VALID`, `EXPIRING_SOON`, `EXPIRED`, or `UNKNOWN`.
- Error Level Analysis for localized compression artifacts.
- Noise-floor and edge-discontinuity checks for spliced or altered package regions.
- Thermal forensic heatmap generation for visual review.

</details>

<details>
<summary><strong>Example verdict narrative</strong></summary>

```text
Inspection VS-A1B2C3D4
Status: WARNING
Why:
  - Expiry date is valid but close to the shelf-life threshold.
  - Packaging substrate has suspicious compression variance near the date stamp.
Action:
  - Hold for manual review before distribution.
Evidence:
  - Annotated OCR image
  - Thermal forensic heatmap
  - Extracted OCR tokens
```

</details>

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Tesseract OCR installed on your machine
- Optional: Docker and Docker Compose

### Backend

<details open>
<summary><strong>Start the verification API</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.app
```

The API starts at `http://localhost:8000`.

</details>

### Frontend

<details open>
<summary><strong>Start the dashboard</strong></summary>

```bash
cd frontend
npm install
npm run dev
```

The dashboard starts at `http://localhost:5173` and proxies API requests to the backend.

</details>

## Docker

<details open>
<summary><strong>Run the full app in a container</strong></summary>

```bash
docker compose up --build
```

The container serves both the FastAPI backend and the built React frontend at `http://localhost:8000`.

</details>

## API Playground

<details open>
<summary><strong>Health check</strong></summary>

```bash
curl http://localhost:8000/api/health
```

Expected shape:

```json
{
  "status": "ok",
  "service": "VeriSight Engine",
  "tesseract": true
}
```

</details>

<details>
<summary><strong>Verify an uploaded image</strong></summary>

```bash
curl -X POST http://localhost:8000/api/verify \
  -F "file=@/path/to/package.jpg" \
  -F "product_name=Sample Product"
```

</details>

<details>
<summary><strong>Send a webcam-style Base64 frame</strong></summary>

```bash
curl -X POST http://localhost:8000/api/verify-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,...",
    "filename": "webcam_scan.jpg",
    "product_name": "Live Scan"
  }'
```

</details>

<details>
<summary><strong>Verify a benchmark sample</strong></summary>

```bash
curl http://localhost:8000/api/samples?limit=5
```

```bash
curl -X POST http://localhost:8000/api/samples/verify \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"expdate_img_00001.jpg"}'
```

</details>

<details>
<summary><strong>Read inspection telemetry</strong></summary>

```bash
curl http://localhost:8000/api/history
curl http://localhost:8000/api/metrics
```

</details>

For the complete endpoint contract, see [docs/api.md](docs/api.md).

## Inspection Report

The API returns a single report object designed to be useful for a UI, an audit log, or a downstream rules engine.

<details open>
<summary><strong>Report anatomy</strong></summary>

```mermaid
mindmap
  root((Inspection report))
    verdict
      status
      reasons
      action
      overall_score
    expiry_analysis
      expiry_status
      expiry_date
      days_remaining
      batch_number
    tamper_analysis
      tamper_score
      authenticity_score
      heatmap
    ocr_analysis
      text_summary
      token_boxes
    images
      raw
      annotated
      heatmap
```

</details>

<details>
<summary><strong>Status cards</strong></summary>

| Status | Meaning | Typical next move |
|---|---|---|
| `PASS` | Date evidence is valid and packaging appears authentic | Approve for consumption or distribution |
| `WARNING` | Date is unclear, close to expiry, or forensics are suspicious | Route to manual review |
| `REJECT` | Product is expired or tampering is detected | Quarantine and investigate |

</details>

<details>
<summary><strong>Minimal response shape</strong></summary>

```json
{
  "id": "VS-A1B2C3D4",
  "verdict": {
    "status": "PASS",
    "reasons": [
      "Authentic packaging substrate",
      "Fresh & Valid"
    ]
  },
  "expiry_analysis": {
    "expiry_status": "VALID",
    "expiry_date": "2027-05-20"
  },
  "tamper_analysis": {
    "status": "AUTHENTIC",
    "tamper_score": 0.058
  }
}
```

</details>

## Architecture

```mermaid
flowchart TB
    subgraph Client[Interactive client]
        UI[React + Vite dashboard]
        Camera[Webcam capture]
        Upload[Drag and drop upload]
        Samples[Dataset explorer]
    end

    subgraph API[FastAPI service]
        Routes[/API routes/]
        Verify[VerificationService]
        History[In-memory history]
    end

    subgraph Engine[Computer vision engine]
        OCR[OCREngine]
        Dates[DateExtractor]
        Tamper[PackagingTamperDetector]
        Render[Annotation renderer]
    end

    UI --> Camera
    UI --> Upload
    UI --> Samples
    Camera --> Routes
    Upload --> Routes
    Samples --> Routes
    Routes --> Verify
    Verify --> OCR
    Verify --> Dates
    Verify --> Tamper
    Verify --> Render
    Verify --> History
    Render --> UI
```

| Layer | Path | Responsibility |
|---|---|---|
| Frontend | `frontend/src/App.jsx` | Uploads, webcam capture, sample browser, visual review, report export |
| API | `backend/api/routes.py` | `/api/verify`, `/api/verify-base64`, `/api/samples`, `/api/history`, `/api/metrics` |
| Orchestration | `backend/services/verification_service.py` | Runs OCR, date extraction, tamper analysis, verdict synthesis |
| OCR/date AI | `backend/ml/ocr_engine.py`, `backend/ml/date_extractor.py` | Text extraction, token boxes, expiry and batch parsing |
| Forensics | `backend/ml/tamper_detector.py` | ELA, noise variance, edge checks, heatmap generation |
| Datasets | `datasets/processed`, `scripts/` | Benchmark manifest creation, splitting, and inspection |

## Verdict Logic

| Tamper status | Expiry status | Verdict | Action |
|---|---|---|---|
| `TAMPERED` | Any | `REJECT` | Quarantine product and review packaging integrity |
| `AUTHENTIC` | `EXPIRED` | `REJECT` | Do not consume or distribute |
| `SUSPICIOUS` | `VALID` | `WARNING` | Send for manual inspection |
| `AUTHENTIC` | `EXPIRING_SOON` | `WARNING` | Prioritize clearance |
| `AUTHENTIC` | `VALID` | `PASS` | Approve for distribution or consumption |

<details>
<summary><strong>Decision flow</strong></summary>

```mermaid
flowchart TD
    A[Inspection complete] --> B{Tampered?}
    B -->|Yes| C[REJECT]
    B -->|No| D{Expired?}
    D -->|Yes| C
    D -->|No| E{Suspicious or expiring soon?}
    E -->|Yes| F[WARNING]
    E -->|No| G{Date detected?}
    G -->|No| F
    G -->|Yes| H[PASS]
```

</details>

## Datasets

VeriSight is prepared for these benchmark sources:

- ExpDate-Real for product packaging date annotations.
- IMD2020 for image manipulation and mask-based tamper evaluation.
- FoodPackagingOCR for packaging text detection and recognition.
- OpenFoodFacts for real commercial product package imagery.

```bash
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python scripts/split_dataset.py
```

More detail lives in [docs/dataset.md](docs/dataset.md).

<details>
<summary><strong>Dataset workflow</strong></summary>

```mermaid
flowchart LR
    A[Raw datasets] --> B[inspect_dataset.py]
    B --> C[prepare_dataset.py]
    C --> D[benchmark_manifest.json]
    D --> E[split_dataset.py]
    E --> F[train / val / test splits]
    F --> G[training and evaluation]
```

</details>

## Training And Evaluation

```bash
python scripts/train_tamper_model.py
pytest tests/test_pipeline.py -v
```

The repository includes serialized model artifacts in `models/` and a generated report at `reports/tamper_model_report.json`.

## Project Map

```text
verisight/
├── backend/                 FastAPI app, routes, services, ML modules
├── frontend/                React + Vite dashboard
├── datasets/processed/      Benchmark manifests and splits
├── docs/                    API, architecture, and dataset notes
├── models/                  Serialized tamper classifier artifacts
├── reports/                 Model and benchmark reports
├── scripts/                 Dataset preparation and training utilities
└── tests/                   Pipeline tests
```

## Docs

- [API reference](docs/api.md)
- [Architecture document](docs/architecture.md)
- [Dataset documentation](docs/dataset.md)

## Try Next

- Run a package image through the dashboard and compare `annotated` vs `heatmap`.
- Call `/api/metrics` after several scans to see pass, warning, and reject counts change.
- Add real files under the raw dataset folders, rebuild the manifest, and inspect new samples from the dashboard.
- Train or replace the tamper model artifacts in `models/`, then rerun the pipeline tests.

## Notes

- Tesseract availability is exposed by `/api/health`.
- The backend keeps recent inspection history in memory, so history resets when the process restarts.
- Docker builds the React app first, then serves the compiled frontend through the FastAPI runtime image.
