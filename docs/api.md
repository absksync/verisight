# VeriSight REST API Reference

All endpoints are hosted at `/api`.

### 1. `POST /api/verify`
Upload a product image for full verification.

**Request:** `multipart/form-data`
- `file`: Image binary (JPEG, PNG, WEBP)
- `product_name`: Optional string name

**Response:**
```json
{
  "id": "VS-A1B2C3D4",
  "filename": "milk_carton.jpg",
  "timestamp": "2026-08-14 12:00:00",
  "processing_time_seconds": 0.42,
  "verdict": {
    "status": "PASS",
    "color": "emerald",
    "action": "APPROVE FOR DISTRIBUTION / CONSUMPTION",
    "reasons": [
      "Authentic packaging substrate (94.2% confidence)",
      "Fresh & Valid. 279 days remaining (Expires: 2027-05-20)"
    ],
    "overall_score": 97.1
  },
  "expiry_analysis": {
    "expiry_status": "VALID",
    "expiry_date": "2027-05-20",
    "days_remaining": 279,
    "mfg_date": "2026-01-10",
    "batch_number": "B77192"
  },
  "tamper_analysis": {
    "status": "AUTHENTIC",
    "tamper_score": 0.058,
    "authenticity_score": 94.2,
    "verdict_description": "Packaging exhibits uniform compression substrate."
  },
  "images": {
    "raw": "data:image/jpeg;base64,...",
    "annotated": "data:image/jpeg;base64,...",
    "heatmap": "data:image/jpeg;base64..."
  }
}
```

---

### 2. `POST /api/verify-base64`
Inspect an image provided as a Base64 URI (useful for webcam frames).

**Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "filename": "webcam_scan.jpg",
  "product_name": "Scanned Product"
}
```

---

### 3. `GET /api/samples`
Returns list of pre-indexed benchmark samples from `ExpDate-Real`, `IMD2020`, and `FoodPackagingOCR`.

---

### 4. `POST /api/samples/verify`
Directly triggers verification on a catalog sample item.

**Request Body:**
```json
{
  "sample_id": "expdate_img_00001.jpg"
}
```

---

### 5. `GET /api/history`
Returns recent inspection logs.

---

### 6. `GET /api/metrics`
Returns system operational metrics (total inspected, pass rate %, avg latency).
