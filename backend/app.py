from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import uvicorn
from backend.config import PROJECT_NAME, VERSION, BASE_DIR
from backend.api.routes import router

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="AI-driven Expiry Date Verification, Forensic Packaging Tamper Detection & Integrity Engine."
)

# Enable CORS for local dev and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount frontend build if it exists
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting {PROJECT_NAME} server on http://localhost:{port}")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=True)
