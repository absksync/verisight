# 🚀 VeriSight — AI Fraud Detection for Product Authenticity

VeriSight is an AI-powered fraud detection system that analyzes product images to detect:

* 🧠 AI-generated or manipulated images
* 📅 Expiry date inconsistencies
* 💰 Price extraction & fraud loss prevention
* 🔍 OCR confidence issues
* 🧬 Visual anomalies & synthetic patterns

Built for real-time verification in logistics, e-commerce returns, and supply chain audits.

---

## ✨ Key Features

### 🔎 Image Fraud Analysis

Upload a product image and VeriSight will:

* Extract expiry date via OCR
* Detect AI manipulation patterns
* Analyze texture / layout inconsistencies
* Generate fraud risk score (0–100)
* Provide explainability tags

---

### 💰 Fraud Savings Engine

If a fraudulent product is detected:

* Product price is extracted automatically
* Refund amount is tracked
* Dashboard shows **total money saved from fraud**

---

### 📊 Live Analytics Dashboard

Backend exposes real statistics:

* Total requests processed
* Fraud cases detected
* Amount saved from fake returns
* Queue length monitoring

No hardcoded values — everything updates dynamically.

---

## 🏗️ Tech Stack

**Frontend**

* Vite + React (or Vanilla JS depending on your setup)
* Tailwind / custom UI
* Live fraud analytics dashboard

**Backend**

* FastAPI
* PyTorch + Vision Transformer
* Tesseract OCR
* Pandas fraud fusion scoring

**ML Pipeline**

* OCR confidence scoring
* Vision Transformer anomaly detection
* Fusion model combining signals

---

## 🚀 How to Run Locally

### 1️⃣ Backend

```bash
cd return0
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Backend runs at:

```
http://localhost:8000
```

---

### 2️⃣ Frontend

```bash
cd verisight-frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## 🔗 API Endpoints

### `POST /predict`

Upload image + delivery date

Returns:

```json
{
  "score": 72,
  "decision": "Reject",
  "timeline": "Expired before delivery",
  "expiry_text": "JAN 2024",
  "ocr_confidence": 0.82,
  "fusion_confidence": 0.76,
  "vit_score": 0.91,
  "tags": "Synthetic Pattern; OCR Confidence Drop",
  "detected_price": 175,
  "refund_saved": 175,
  "queue_position": null
}
```

---

### `GET /stats`

Returns dashboard analytics:

```json
{
  "total_requests": 12,
  "rejected": 4,
  "saved_amount": 870,
  "queue_length": 0
}
```

---

## 🎯 Use Cases

* E-commerce return fraud detection
* Warehouse verification
* Supply chain inspection
* Insurance claim validation
* Anti-counterfeit monitoring

---

## 🧠 Future Improvements

* Real-time video inspection
* Blockchain audit trail
* Retail POS integration
* Mobile scanning app
* Federated model learning

---

## 👨‍💻 Team VeriSight

Built for hackathons, real-world fraud prevention, and production-ready deployment.

---

⭐ If you like this project, give it a star — it helps a lot!
