"""
app/main.py — MpoxNet-V FastAPI Application
============================================
Monkeypox Skin Lesion Classification API
  POST /predict         — classify image
  POST /predict/batch   — classify multiple images
  POST /predict/attention — attention rollout map
  GET  /predict/classes  — class labels
  POST /train/start      — start 5-fold training
  GET  /train/status     — live training status
  GET  /train/results    — final results
  POST /train/load       — load checkpoint
  GET  /health           — health check
  GET  /                 — web UI demo
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.routers import predict, train, health
from models.inference import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    logger.info("MpoxNet-V API starting up...")
    model_path = os.getenv("MODEL_PATH", "./saved_models/mpoxnet_v_best.pt")
    model_manager.load(model_path=model_path if Path(model_path).exists() else None)
    yield
    # ── Shutdown ──────────────────────────────────────────────
    logger.info("MpoxNet-V API shutting down.")


app = FastAPI(
    title="MpoxNet-V API",
    description=(
        "Monkeypox Skin Lesion Classification using a Dual-Branch "
        "Vision Transformer (DeiT-B + EfficientNetB4) with "
        "Cross-Attention Gate Fusion.\n\n"
        "**Dataset:** MSLD v2.0 — 755 images, 541 patients, 6 classes, dermatologist-approved.\n\n"
        "**Baseline:** SwinTransformer 93.71% (Vuran et al., 2025).\n\n"
        "**Reference paper:** MpoxNet-V Research Proposal (2024)."
    ),
    version="1.0.0",
    contact={"name": "MpoxNet-V Research Team"},
    lifespan=lifespan,
)

# CORS — allow all origins for demo; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(health.router)

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h2>MpoxNet-V API running. Visit <a href='/docs'>/docs</a></h2>")
