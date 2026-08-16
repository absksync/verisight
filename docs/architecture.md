# VeriSight Architecture Document

## System Architecture

VeriSight is structured into three decoupled layers:
1. **Core Forensic & Computer Vision Engine (Python/OpenCV/Tesseract)**
2. **API & Orchestration Service (FastAPI)**
3. **Interactive Inspection Client (React / Vite)**

```
+-------------------------------------------------------------------+
|                        Interactive Frontend                       |
|        (Drag & Drop / WebCam / Dataset Explorer / Heatmap)        |
+---------------------------------+---------------------------------+
                                  | HTTP / REST
+---------------------------------v---------------------------------+
|                       FastAPI Routing Layer                       |
|               (/verify, /samples, /history, /metrics)             |
+---------------------------------+---------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
+-----------v-----------+                   +-----------v-----------+
|    OCREngine & Date   |                   | Forensic TamperEngine |
|   Extraction Pipeline |                   |  (ELA, Noise, Edge)   |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
+---------------------------------v---------------------------------+
|                    Verification Orchestrator                      |
|      - Cross-validation & Shelf-life delta calculation            |
|      - High-contrast visual annotation rendering                  |
|      - PASS / WARNING / REJECT Decision Engine                    |
+-------------------------------------------------------------------+
```

---

## 1. OCR & Date Parsing Pipeline

- **CLAHE Enhancement**: Standardizes dynamic range across specular highlights on plastic wrappers and aluminum cans.
- **Tesseract OCR (PSM 11 / PSM 6)**: Extracts text tokens and bounding coordinates.
- **Contextual Date Parsing**: Recognizes standard ISO, slash, dot, alpha-month, and compact date expressions.
- **Semantic Classification**: Disambiguates manufacturing date vs expiration date using surrounding keyword tokens (`EXP`, `BB`, `USE BY`, `MFG`, `PKD`).

---

## 2. Forensic Packaging Integrity Detection

- **Error Level Analysis (ELA)**: Detects localized compression artifacts caused by pasting digital characters over packaging.
- **Laplacian Noise Floor Variance**: Identifies discrepancies in camera sensor noise.
- **Gradient Discontinuity**: Detects unnatural boundary falloffs around modified dates or barcodes.
- **Thermal Heatmap Synthesis**: Combines ELA, noise, and edge maps into a thermal Turbo colormap overlay.

---

## 3. Decision Matrix

| Tamper Status | Expiry Status | Overall Verdict | Recommended Action |
|---|---|---|---|
| **TAMPERED** | *Any* | **`REJECT`** | Quarantine product; packaging manipulation detected |
| **AUTHENTIC** | **EXPIRED** | **`REJECT`** | Do not consume or distribute; product expired |
| **SUSPICIOUS** | **VALID** | **`WARNING`** | Manual inspection required; noise discrepancy |
| **AUTHENTIC** | **EXPIRING_SOON** | **`WARNING`** | Prioritize clearance; expires in < 30 days |
| **AUTHENTIC** | **VALID** | **`PASS`** | Approved for consumption and retail distribution |
