"""app/routers/health.py — Health check endpoint"""
from fastapi import APIRouter
from models.inference import model_manager, CLASSES

router = APIRouter(tags=["Health"])

@router.get("/health", summary="API health check")
async def health():
    params = model_manager.model.count_parameters() if model_manager.model else 0
    return {
        "status":         "ok",
        "model_loaded":   model_manager.loaded,
        "model_version":  "MpoxNet-V 1.0",
        "device":         str(model_manager.device),
        "classes":        CLASSES,
        "num_parameters": f"{params:,}",
        "architecture":   "DeiT-B + EfficientNetB4 + CrossAttentionGate",
        "dataset":        "MSLD v2.0 (755 images, 541 patients, 6 classes)",
    }
