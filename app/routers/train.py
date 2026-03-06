"""
app/routers/train.py
Endpoints:
  POST /train/start   — start 5-fold cross-validation training
  GET  /train/status  — live training status
  GET  /train/results — final results after training completes
  POST /train/load    — load a saved .pt checkpoint
"""
import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import TrainingConfig, TrainingStatus
from models.trainer import Trainer, TRAINING_STATE
from models.inference import model_manager

router = APIRouter(prefix="/train", tags=["Training"])
logger = logging.getLogger(__name__)
SAVE_DIR = Path("./saved_models")
SAVE_DIR.mkdir(exist_ok=True)


def _run_training(cfg: dict):
    """Background task: runs full 5-fold training."""
    try:
        from utils.dataset import get_kfold_loaders, compute_class_weights
        from sklearn.metrics import f1_score, accuracy_score
        import numpy as np

        TRAINING_STATE.status     = "running"
        TRAINING_STATE.total_folds = cfg.get("num_folds", 5)
        TRAINING_STATE.fold_results = []

        data_dir = cfg.get("data_dir", "./data/MSLD_v2")
        if not Path(data_dir).exists():
            TRAINING_STATE.status  = "failed"
            TRAINING_STATE.message = f"Data directory not found: {data_dir}"
            return

        fold_loaders, base_dataset = get_kfold_loaders(
            data_dir,
            n_splits=cfg.get("num_folds", 5),
            batch_size=cfg.get("batch_size", 16),
        )
        class_weights = compute_class_weights(base_dataset)

        trainer      = Trainer(cfg)
        fold_results = []

        for fold_num, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
            TRAINING_STATE.current_fold = fold_num
            save_path = str(SAVE_DIR / f"mpoxnet_v_fold{fold_num}.pt")
            result    = trainer.train_fold(
                train_loader, val_loader,
                fold_num=fold_num,
                class_weights=class_weights,
                save_path=save_path,
            )
            fold_results.append(result["metrics"])
            TRAINING_STATE.fold_results.append({
                "fold":     fold_num,
                "accuracy": result["metrics"]["accuracy"],
                "f1":       result["metrics"]["f1"],
                "kappa":    result["metrics"]["kappa"],
                "auc":      result["metrics"]["auc"],
            })

        # Aggregate
        accs   = [r["accuracy"] for r in fold_results]
        f1s    = [r["f1"]       for r in fold_results]
        kappas = [r["kappa"]    for r in fold_results]
        aucs   = [r["auc"]      for r in fold_results if r["auc"] is not None]

        TRAINING_STATE.final_results = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1":       float(np.mean(f1s)),
            "std_f1":        float(np.std(f1s)),
            "mean_kappa":    float(np.mean(kappas)),
            "std_kappa":     float(np.std(kappas)),
            "mean_auc":      float(np.mean(aucs)) if aucs else None,
            "std_auc":       float(np.std(aucs))  if aucs else None,
            "fold_results":  TRAINING_STATE.fold_results,
            "best_fold_model": str(SAVE_DIR / f"mpoxnet_v_fold{int(np.argmax(f1s))+1}.pt"),
        }
        TRAINING_STATE.status  = "completed"
        TRAINING_STATE.message = (
            f"Training complete. "
            f"Mean Acc: {np.mean(accs):.4f} | Mean F1: {np.mean(f1s):.4f}"
        )
        logger.info(TRAINING_STATE.message)

        # Auto-load best fold model
        best_path = TRAINING_STATE.final_results["best_fold_model"]
        model_manager.load(best_path, num_classes=cfg.get("num_classes", 6))

    except Exception as e:
        TRAINING_STATE.status  = "failed"
        TRAINING_STATE.message = str(e)
        logger.error(f"Training failed: {e}", exc_info=True)


@router.post("/start", summary="Start 5-fold cross-validation training")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Launches training in the background. Poll /train/status for progress.
    Expects MSLD v2.0 data at config.data_dir with subfolders per class.
    """
    if TRAINING_STATE.status == "running":
        raise HTTPException(status_code=409, detail="Training already running.")

    TRAINING_STATE.status        = "starting"
    TRAINING_STATE.current_phase = 0
    TRAINING_STATE.current_epoch = 0
    TRAINING_STATE.fold_results  = []
    TRAINING_STATE.final_results = {}

    background_tasks.add_task(_run_training, config.model_dump())
    return {
        "message":    "Training started in background.",
        "status_url": "/train/status",
        "config":     config.model_dump(),
    }


@router.get("/status", response_model=TrainingStatus, summary="Live training status")
async def training_status():
    """Poll this endpoint to track training progress."""
    return TrainingStatus(
        status        = TRAINING_STATE.status,
        current_phase = TRAINING_STATE.current_phase,
        current_epoch = TRAINING_STATE.current_epoch,
        total_epochs  = TRAINING_STATE.total_epochs,
        current_fold  = TRAINING_STATE.current_fold,
        total_folds   = TRAINING_STATE.total_folds,
        val_accuracy  = TRAINING_STATE.val_accuracy,
        val_f1        = TRAINING_STATE.val_f1,
        best_f1       = TRAINING_STATE.best_f1,
        message       = TRAINING_STATE.message,
    )


@router.get("/results", summary="Final training results (after completion)")
async def training_results():
    """Returns aggregated 5-fold results once training is complete."""
    if TRAINING_STATE.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Training not complete. Current status: {TRAINING_STATE.status}"
        )
    return {
        "status":  TRAINING_STATE.status,
        "results": TRAINING_STATE.final_results,
    }


@router.post("/load", summary="Load a saved model checkpoint")
async def load_model(model_path: str, num_classes: int = 6):
    """Load a previously saved .pt checkpoint file into the inference engine."""
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {model_path}")
    ok = model_manager.load(model_path, num_classes=num_classes)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to load model.")
    return {"message": f"Model loaded from {model_path}", "device": str(model_manager.device)}


@router.get("/saved", summary="List saved model checkpoints")
async def list_saved():
    """List all .pt files in the saved_models directory."""
    files = list(SAVE_DIR.glob("*.pt"))
    return {
        "saved_models": [
            {"path": str(f), "size_mb": round(f.stat().st_size / 1e6, 2)}
            for f in files
        ]
    }
