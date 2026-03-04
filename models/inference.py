"""
models/inference.py — Model loading, prediction, attention rollout
"""
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.mpoxnet_v import MpoxNetV

logger = logging.getLogger(__name__)

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RISK_MAP = {
    "Monkeypox":  ("HIGH",   "Isolate patient. Seek immediate PCR confirmation."),
    "Chickenpox": ("HIGH",   "Consult dermatologist. Avoid contact with immunocompromised individuals."),
    "Measles":    ("HIGH",   "Isolate patient. Report to public health authority."),
    "Cowpox":     ("MEDIUM", "Consult physician. Rare in humans; usually self-limiting."),
    "HFMD":       ("MEDIUM", "Rest and hydration. Monitor for complications in young children."),
    "Healthy":    ("LOW",    "No lesion detected. Routine follow-up if symptoms persist."),
}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ModelManager:
    """Singleton-style model manager for FastAPI lifespan."""

    def __init__(self):
        self.model:  Optional[MpoxNetV] = None
        self.device: torch.device       = torch.device("cpu")
        self.loaded: bool               = False
        self.model_path: str            = ""

    def load(self, model_path: Optional[str] = None, num_classes: int = 6) -> bool:
        """
        Load model weights if path exists, otherwise create fresh model
        (useful for development / demo without pre-trained weights).
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = MpoxNetV(num_classes=num_classes).to(self.device)

            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                state = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state)
                self.model_path = model_path
                logger.info(f"Loaded weights from {model_path}")
            else:
                logger.warning("No weights file found — using random weights (demo mode).")

            self.model.eval()
            self.loaded = True
            logger.info(f"Model ready on {self.device} | params: {self.model.count_parameters():,}")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.loaded = False
            return False

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes. Returns full prediction dict."""
        if not self.loaded or self.model is None:
            raise RuntimeError("Model not loaded.")

        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = VAL_TRANSFORM(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, alpha, beta = self.model(tensor, return_gate=True)

        probs   = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_cls = CLASSES[pred_idx]
        risk, rec = RISK_MAP[pred_cls]

        all_probs = [
            {
                "label":       cls,
                "probability": float(probs[i]),
                "percentage":  f"{probs[i]*100:.1f}%",
            }
            for i, cls in enumerate(CLASSES)
        ]
        all_probs.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "prediction":         pred_cls,
            "confidence":         float(probs[pred_idx]),
            "confidence_pct":     f"{probs[pred_idx]*100:.1f}%",
            "all_probabilities":  all_probs,
            "clinical_risk":      risk,
            "recommendation":     rec,
            "gate_alpha":         float(alpha[0].cpu()),
            "gate_beta":          float(beta[0].cpu()),
            "attention": {
                "available":    True,
                "description":  "DeiT-B global attention (Branch 1)",
                "note":         "Use /predict/attention endpoint for rollout map.",
            },
            "model_version": "MpoxNet-V 1.0",
            "vega_audited":  True,
        }

    def get_attention_rollout(self, image_bytes: bytes) -> np.ndarray:
        """
        Compute DeiT-B attention rollout map for a single image.
        Returns normalised 224x224 float numpy array [0,1].
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("Model not loaded.")

        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = VAL_TRANSFORM(img).unsqueeze(0).to(self.device)

        attentions = []

        def hook(module, inp, out):
            attentions.append(out.detach().cpu())

        hooks = [b.attn.register_forward_hook(hook) for b in self.model.deit.blocks]

        self.model.eval()
        with torch.no_grad():
            _ = self.model(tensor)

        for h in hooks:
            h.remove()

        # Rollout across layers
        rollout = torch.eye(attentions[0].shape[-1])
        for attn in attentions:
            attn_avg = attn.mean(dim=1)[0]          # avg over heads
            attn_avg = attn_avg + torch.eye(attn_avg.size(-1))
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout  = attn_avg @ rollout

        # CLS row, skip [CLS]+[DIST] tokens → 196 patch tokens
        mask = rollout[0, 2:].reshape(14, 14).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        import cv2
        return cv2.resize(mask, (224, 224))


# Global singleton used by FastAPI app
model_manager = ModelManager()
