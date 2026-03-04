"""
app/routers/predict.py
Endpoints:
  POST /predict          — single image classification
  POST /predict/batch    — multiple images
  GET  /predict/attention — attention rollout map (base64 PNG)
  GET  /predict/classes   — list class labels
"""
import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.schemas import PredictionResponse
from models.inference import model_manager, CLASSES

logger = APIRouter()
router = APIRouter(prefix="/predict", tags=["Prediction"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def _check_file(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {file.content_type}. Use JPEG/PNG.")


@router.post("", response_model=PredictionResponse, summary="Classify a skin lesion image")
async def predict(file: UploadFile = File(..., description="Skin lesion image (JPEG/PNG)")):
    """
    Upload a skin lesion image and get:
    - Predicted class (Monkeypox, Chickenpox, Measles, Cowpox, HFMD, Healthy)
    - Confidence score and all class probabilities
    - Clinical risk level and recommended action
    - Gate weights (alpha = DeiT-B, beta = EfficientNetB4)
    """
    _check_file(file)
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /health to check.")

    try:
        image_bytes = await file.read()
        result      = model_manager.predict(image_bytes)
        return result
    except Exception as e:
        logging.getLogger(__name__).error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", summary="Classify multiple images")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify up to 20 images in one request. Returns list of predictions + summary."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images per batch.")
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    summary = {cls: 0 for cls in CLASSES}

    for f in files:
        _check_file(f)
        try:
            bts    = await f.read()
            result = model_manager.predict(bts)
            results.append({"filename": f.filename, **result})
            summary[result["prediction"]] = summary.get(result["prediction"], 0) + 1
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"total_images": len(files), "results": results, "summary": summary}


@router.post("/attention", summary="Get attention rollout map as base64 PNG")
async def attention_map(file: UploadFile = File(...)):
    """
    Returns a base64-encoded PNG showing where DeiT-B is attending.
    Overlay this on the original image to validate lesion focus.
    """
    _check_file(file)
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        image_bytes = await file.read()
        rollout     = model_manager.get_attention_rollout(image_bytes)

        # Convert to heatmap PNG
        import cv2
        heatmap = (rollout * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        _, buf  = cv2.imencode(".png", heatmap)
        b64     = base64.b64encode(buf.tobytes()).decode()

        return {
            "attention_map_base64": b64,
            "format":  "PNG",
            "size":    "224x224",
            "note":    "Jet heatmap of DeiT-B attention rollout. Red = high attention.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classes", summary="List all class labels and clinical info")
async def get_classes():
    """Returns the 6 MSLD v2.0 class labels with clinical context."""
    return {
        "classes": [
            {"id": 0, "name": "Monkeypox",  "risk": "HIGH",   "dataset_count": 284},
            {"id": 1, "name": "Chickenpox", "risk": "HIGH",   "dataset_count": 75},
            {"id": 2, "name": "Measles",    "risk": "HIGH",   "dataset_count": 55},
            {"id": 3, "name": "Cowpox",     "risk": "MEDIUM", "dataset_count": 66},
            {"id": 4, "name": "HFMD",       "risk": "MEDIUM", "dataset_count": 161},
            {"id": 5, "name": "Healthy",    "risk": "LOW",    "dataset_count": 114},
        ],
        "dataset": "MSLD v2.0 (Ali et al., 2024) — 755 images, 541 patients, dermatologist-approved",
        "total_images": 755,
    }
