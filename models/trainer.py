"""
models/trainer.py — Three-phase training pipeline for MpoxNet-V
"""
import logging
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

from models.mpoxnet_v import MpoxNetV, CombinedLoss

logger = logging.getLogger(__name__)

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]


# ─── Shared training state (read by /train/status endpoint) ──────────────────

@dataclass
class TrainingState:
    status:        str   = "idle"
    current_phase: int   = 0
    current_epoch: int   = 0
    total_epochs:  int   = 0
    current_fold:  int   = 0
    total_folds:   int   = 5
    val_accuracy:  float = 0.0
    val_f1:        float = 0.0
    best_f1:       float = 0.0
    message:       str   = ""
    fold_results:  list  = field(default_factory=list)
    final_results: dict  = field(default_factory=dict)

TRAINING_STATE = TrainingState()


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def cutmix(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = imgs.shape
    ri = torch.randperm(B)
    cr = np.sqrt(1 - lam)
    cw, ch = int(W * cr), int(H * cr)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(0, cx - cw // 2), min(W, cx + cw // 2)
    y1, y2 = max(0, cy - ch // 2), min(H, cy + ch // 2)
    imgs_m = imgs.clone()
    imgs_m[:, :, y1:y2, x1:x2] = imgs[ri, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return imgs_m, labels, labels[ri], lam

def mixup(imgs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    ri  = torch.randperm(imgs.size(0))
    return lam * imgs + (1 - lam) * imgs[ri], labels, labels[ri], lam


# ─── Per-epoch helpers ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, cutmix_p=0.3, mixup_p=0.2):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        r = np.random.rand()
        if r < cutmix_p:
            imgs, la, lb, lam = cutmix(imgs, labels)
            loss = lam * criterion(model(imgs), la) + (1 - lam) * criterion(model(imgs), lb)
        elif r < cutmix_p + mixup_p:
            imgs, la, lb, lam = mixup(imgs, labels)
            loss = lam * criterion(model(imgs), la) + (1 - lam) * criterion(model(imgs), lb)
        else:
            loss = criterion(model(imgs), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, labels, probs_all = [], [], []
    for imgs, lbls in loader:
        out   = model(imgs.to(device))
        p     = torch.softmax(out, 1).cpu().numpy()
        probs_all.extend(p)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs_all, multi_class="ovr", average="macro")
    except Exception:
        auc = None
    return {"accuracy": acc, "f1": f1, "kappa": kappa, "auc": auc,
            "preds": preds, "labels": labels, "probs": probs_all}


# ─── Three-phase trainer ──────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict, on_epoch_end: Optional[Callable] = None):
        self.cfg          = cfg
        self.on_epoch_end = on_epoch_end
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _new_model(self):
        return MpoxNetV(
            num_classes=self.cfg.get("num_classes", 6),
            dropout=self.cfg.get("dropout", 0.35),
        ).to(self.device)

    def _criterion(self, weights=None):
        if weights is not None:
            weights = weights.to(self.device)
        return CombinedLoss(
            class_weights=weights,
            gamma=self.cfg.get("focal_gamma", 2.0),
            label_smoothing=self.cfg.get("label_smoothing", 0.1),
        )

    def _log(self, msg):
        TRAINING_STATE.message = msg
        logger.info(msg)
        if self.on_epoch_end:
            self.on_epoch_end(TRAINING_STATE)

    def train_fold(self, train_loader, val_loader, fold_num, class_weights=None, save_path=None):
        model     = self._new_model()
        criterion = self._criterion(class_weights)
        cfg       = self.cfg
        ph1       = cfg.get("phase1_epochs", 5)
        ph2       = cfg.get("phase2_epochs", 13)
        ph3       = cfg.get("phase3_epochs", 42)
        patience  = cfg.get("patience", 12)
        lr_head   = cfg.get("learning_rate", 1e-3)
        lr_bb     = lr_head * 0.05

        TRAINING_STATE.total_epochs = ph1 + ph2 + ph3
        TRAINING_STATE.current_fold = fold_num
        best_f1, best_state, wait = 0.0, None, 0
        epoch_g = 0

        # ── Phase 1: gate + head only ─────────────────────────────────
        TRAINING_STATE.current_phase = 1
        model.freeze_backbones()
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr_head, weight_decay=cfg.get("weight_decay", 0.01))
        for ep in range(1, ph1 + 1):
            epoch_g += 1
            TRAINING_STATE.current_epoch = epoch_g
            t0   = time.time()
            loss = train_epoch(model, train_loader, opt, criterion, self.device)
            m    = eval_epoch(model, val_loader, self.device)
            TRAINING_STATE.val_accuracy, TRAINING_STATE.val_f1 = m["accuracy"], m["f1"]
            self._log(f"[Fold {fold_num}] Ph1 Ep {ep}/{ph1} | loss={loss:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} ({time.time()-t0:.1f}s)")

        # ── Phase 2: partial unfreeze ──────────────────────────────────
        TRAINING_STATE.current_phase = 2
        model.unfreeze_last_layers()
        opt = optim.AdamW([
            {"params": model.deit.parameters(),   "lr": lr_bb},
            {"params": model.effnet.parameters(),  "lr": lr_bb},
            {"params": list(model.proj_g.parameters()) + list(model.proj_l.parameters())
                     + list(model.gate.parameters()) + list(model.head.parameters()),
             "lr": lr_head * 0.1},
        ], weight_decay=cfg.get("weight_decay", 0.01))
        sched = CosineAnnealingWarmRestarts(opt, T_0=6, T_mult=2)
        wait  = 0
        for ep in range(1, ph2 + 1):
            epoch_g += 1
            TRAINING_STATE.current_epoch = epoch_g
            t0 = time.time()
            loss = train_epoch(model, train_loader, opt, criterion, self.device)
            m    = eval_epoch(model, val_loader, self.device)
            sched.step()
            TRAINING_STATE.val_accuracy, TRAINING_STATE.val_f1 = m["accuracy"], m["f1"]
            if m["f1"] > best_f1:
                best_f1, best_state, wait = m["f1"], {k: v.clone() for k, v in model.state_dict().items()}, 0
                TRAINING_STATE.best_f1 = best_f1
            else:
                wait += 1
            self._log(f"[Fold {fold_num}] Ph2 Ep {ep}/{ph2} | loss={loss:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} best={best_f1:.4f} ({time.time()-t0:.1f}s)")
            if wait >= patience:
                logger.info(f"Early stop Phase 2 at epoch {ep}")
                break

        # ── Phase 3: full fine-tuning ──────────────────────────────────
        TRAINING_STATE.current_phase = 3
        model.unfreeze_all()
        opt   = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=cfg.get("weight_decay", 0.01))
        sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
        wait  = 0
        for ep in range(1, ph3 + 1):
            epoch_g += 1
            TRAINING_STATE.current_epoch = min(epoch_g, TRAINING_STATE.total_epochs)
            t0   = time.time()
            loss = train_epoch(model, train_loader, opt, criterion, self.device)
            m    = eval_epoch(model, val_loader, self.device)
            sched.step()
            TRAINING_STATE.val_accuracy, TRAINING_STATE.val_f1 = m["accuracy"], m["f1"]
            if m["f1"] > best_f1:
                best_f1, best_state, wait = m["f1"], {k: v.clone() for k, v in model.state_dict().items()}, 0
                TRAINING_STATE.best_f1 = best_f1
            else:
                wait += 1
            self._log(f"[Fold {fold_num}] Ph3 Ep {ep}/{ph3} | loss={loss:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} best={best_f1:.4f} ({time.time()-t0:.1f}s)")
            if wait >= patience:
                logger.info(f"Early stop Phase 3 at epoch {ep}")
                break

        if best_state:
            model.load_state_dict(best_state)

        final = eval_epoch(model, val_loader, self.device)

        if save_path:
            torch.save({"model_state_dict": model.state_dict(),
                        "num_classes": self.cfg.get("num_classes", 6),
                        "fold": fold_num, "best_f1": best_f1, "metrics": final}, save_path)
            logger.info(f"Saved: {save_path}")

        return {"metrics": final, "best_f1": best_f1}
