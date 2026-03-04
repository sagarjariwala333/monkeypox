"""
models/mpoxnet_v.py — MpoxNet-V Architecture
=============================================
Dual-Branch Vision Transformer for Monkeypox Skin Lesion Classification

  Branch 1  DeiT-B (distilled ViT)   -> 1536-D global attention features
  Branch 2  EfficientNetB4            -> 1792-D local texture features
  Fusion    Cross-Attention Gate      -> learned alpha, beta per image
  Head      Linear 1024->512->6

Design choices:
  - DeiT-B over plain ViT: distillation token transfers CNN inductive
    biases, resolving small-dataset weakness [Al-Hammuri 2023]
  - Cross-attention gate: dynamic per-image global/local weighting
  - Target: surpass SwinTransformer 93.71% on audited MSLD v2.0
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class CrossAttentionGate(nn.Module):
    """Learns alpha,beta weights for dual-branch fusion. alpha+beta=1."""

    def __init__(self, dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, g: torch.Tensor, l: torch.Tensor):
        w = self.gate(torch.cat([g, l], dim=1))  # [B,2]
        a, b = w[:, 0:1], w[:, 1:2]
        return a * g + b * l, a.squeeze(1), b.squeeze(1)


class MpoxNetV(nn.Module):
    FUSION_DIM = 1024

    def __init__(self, num_classes: int = 6, dropout: float = 0.35):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("pip install timm")
        self.num_classes = num_classes

        # Branch 1: DeiT-B — [CLS]+[DIST] = 1536-D
        self.deit = timm.create_model(
            "deit_base_distilled_patch16_224", pretrained=True, num_classes=0
        )
        self.proj_g = nn.Sequential(
            nn.Linear(768 * 2, self.FUSION_DIM),
            nn.BatchNorm1d(self.FUSION_DIM),
            nn.GELU(),
        )

        # Branch 2: EfficientNetB4 — 1792-D
        self.effnet = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        self.proj_l = nn.Sequential(
            nn.Linear(1792, self.FUSION_DIM),
            nn.BatchNorm1d(self.FUSION_DIM),
            nn.GELU(),
        )

        self.gate = CrossAttentionGate(dim=self.FUSION_DIM)

        self.head = nn.Sequential(
            nn.Linear(self.FUSION_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        tokens = self.deit.forward_features(x)          # [B,198,768]
        G = self.proj_g(torch.cat([tokens[:, 0], tokens[:, 1]], dim=1))
        L = self.proj_l(self.effnet(x))
        F, alpha, beta = self.gate(G, L)
        logits = self.head(F)
        return (logits, alpha, beta) if return_gate else logits

    def freeze_backbones(self):
        for p in list(self.deit.parameters()) + list(self.effnet.parameters()):
            p.requires_grad = False

    def unfreeze_last_layers(self):
        for block in list(self.deit.blocks)[-4:]:
            for p in block.parameters(): p.requires_grad = True
        for p in self.deit.norm.parameters(): p.requires_grad = True
        for stage in list(self.effnet.blocks)[-3:]:
            for p in stage.parameters(): p.requires_grad = True
        for p in self.effnet.conv_head.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CombinedLoss(nn.Module):
    """0.7 * CrossEntropy (class-weighted) + 0.3 * Focal Loss (gamma=2)"""

    def __init__(self, class_weights=None, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = self.ce(inputs, targets)
        return 0.7 * ce + 0.3 * ((1 - torch.exp(-ce)) ** self.gamma) * ce
