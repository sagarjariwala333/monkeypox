"""app/schemas.py — Pydantic request/response models"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class SkinClass(str, Enum):
    MONKEYPOX  = "Monkeypox"
    CHICKENPOX = "Chickenpox"
    MEASLES    = "Measles"
    COWPOX     = "Cowpox"
    HFMD       = "HFMD"
    HEALTHY    = "Healthy"

class ClassProb(BaseModel):
    label:       SkinClass
    probability: float
    percentage:  str

class PredictionResponse(BaseModel):
    prediction:          str
    confidence:          float
    confidence_pct:      str
    all_probabilities:   list[ClassProb]
    clinical_risk:       str
    recommendation:      str
    gate_alpha:          Optional[float] = None
    gate_beta:           Optional[float] = None
    attention:           dict
    model_version:       str = "MpoxNet-V 1.0"
    vega_audited:        bool = True

class TrainingConfig(BaseModel):
    num_classes:     int   = Field(default=6,    ge=2,   le=10)
    batch_size:      int   = Field(default=16,   ge=4,   le=64)
    phase1_epochs:   int   = Field(default=5,    ge=1,   le=20)
    phase2_epochs:   int   = Field(default=13,   ge=1,   le=50)
    phase3_epochs:   int   = Field(default=42,   ge=1,   le=100)
    learning_rate:   float = Field(default=1e-3, ge=1e-6, le=0.1)
    weight_decay:    float = Field(default=0.01)
    dropout:         float = Field(default=0.35, ge=0.0, le=0.8)
    focal_gamma:     float = Field(default=2.0,  ge=0.0, le=5.0)
    label_smoothing: float = Field(default=0.1,  ge=0.0, le=0.3)
    patience:        int   = Field(default=12,   ge=3,   le=30)
    data_dir:        str   = Field(default="./data/MSLD_v2")
    num_folds:       int   = Field(default=5,    ge=2,   le=10)
    run_vega_audit:  bool  = Field(default=True)

class TrainingStatus(BaseModel):
    status:        str
    current_phase: int
    current_epoch: int
    total_epochs:  int
    current_fold:  int
    total_folds:   int
    val_accuracy:  Optional[float] = None
    val_f1:        Optional[float] = None
    best_f1:       Optional[float] = None
    message:       str = ""

class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_version:  str
    device:         str
    classes:        list[str]
    num_parameters: str
