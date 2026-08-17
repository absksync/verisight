# ─────────────────────────────────────────────────────────────────────────────
# VeriSight — Expiry & Packaging Integrity AI
# Multi-stage: Node (frontend build) → Python (backend + serve)
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --quiet
COPY frontend/ ./
RUN npm run build


# Stage 2: Python backend + serve everything
FROM python:3.11-slim AS runtime

LABEL maintainer="absksync" \
      description="VeriSight — AI-powered expiry date & packaging tamper detection"

# System dependencies: Tesseract OCR + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libjpeg-dev \
        libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY models/ ./models/
COPY datasets/processed/ ./datasets/processed/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Runtime directories
RUN mkdir -p backend/uploads reports

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
