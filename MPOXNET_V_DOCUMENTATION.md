# MpoxNet-V: Dual-Branch Vision Transformer for Monkeypox Skin Lesion Classification

## A Research Proposal and Technical Documentation

---

## Abstract

This document presents MpoxNet-V, a novel deep learning architecture designed for automated classification of monkeypox and similar skin lesion diseases. The system employs a dual-branch Vision Transformer architecture combining Data-efficient Image Transformer (DeiT-B) with EfficientNetB4, fused through a learned Cross-Attention Gate mechanism. The proposed model targets classification across six disease categories using the Monkeypox Skin Lesion Dataset v2.0 (MSLD v2.0), with the goal of surpassing the current state-of-the-art performance of 93.71% achieved by SwinTransformer. This documentation outlines the implementation methodology, architectural decisions, training strategies, and clinical integration approach suitable for academic study papers.

---

## 1. Introduction

### 1.1 Background

The emergence of monkeypox (Mpox) as a global health concern has highlighted the urgent need for rapid, accurate diagnostic tools. Monkeypox, caused by the monkeypox virus, presents with skin lesions that can be clinically confused with other viral exanthems including chickenpox, measles, cowpox, and Hand-Foot-Mouth Disease (HFMD). Early and accurate differentiation is crucial for appropriate patient management, isolation protocols, and public health response.

Traditional diagnostic methods rely on polymerase chain reaction (PCR) testing, which while accurate, requires laboratory infrastructure and processing time. Clinical visual inspection by dermatologists offers faster initial assessment but is subject to inter-observer variability and may lack sensitivity for distinguishing between similar-appearing lesions.

### 1.2 Problem Statement

The challenge addressed by this research is the development of an automated, computer vision-based system capable of classifying skin lesion images into one of six categories:
- Monkeypox
- Chickenpox
- Measles
- Cowpox
- Hand-Foot-Mouth Disease (HFMD)
- Healthy (no lesion)

This multi-class classification problem is challenging due to:
- Visual similarity between different viral exanthems
- Variations in lesion morphology across disease stages
- Image quality variations in clinical photography
- Class imbalance in available datasets
- Need for interpretable predictions suitable for clinical decision support

### 1.3 Research Objectives

The primary objectives of this research are:

1. **Develop a novel dual-branch architecture** that leverages both global attention mechanisms and local texture analysis for skin lesion classification

2. **Achieve superior classification performance** compared to existing approaches on the MSLD v2.0 dataset, targeting performance above 93.71% (current SwinTransformer SOTA)

3. **Provide clinically meaningful outputs** including prediction confidence, risk stratification, and actionable recommendations

4. **Enable model interpretability** through attention visualization to build clinical trust and facilitate expert review

5. **Create a deployable API** that can be integrated into clinical workflows for real-time inference

---

## 2. Literature Review and Related Work

### 2.1 Existing Approaches

Previous research in automated monkeypox skin lesion classification has employed various deep learning architectures:

| Study | Architecture | Performance | Limitations |
|-------|-------------|-------------|-------------|
| Uysal et al. (2023) | CNN-LSTM | 87% accuracy, Kappa=0.8222 | Limited to CNN features only |
| Jaradat et al. (2023) | MobileNetV2 | 98.16% | Potential artifact issues |
| Vega et al. (2023) | Various | Dataset quality critique | Highlighted dataset limitations |
| Vuran et al. (2025) | SwinTransformer | 93.71% | Current SOTA |

### 2.2 Vision Transformers in Medical Imaging

The application of Vision Transformers (ViT) to medical imaging has shown promising results. The transformer architecture's ability to capture long-range dependencies makes it particularly suitable for analyzing spatial relationships in medical images. However, ViTs typically require large datasets for effective training, which is a limitation in medical imaging domains where annotated data is scarce.

### 2.3 Knowledge Distillation

The Data-efficient Image Transformer (DeiT) addresses the data efficiency limitation by incorporating a distillation token that enables knowledge transfer from a CNN teacher model. This approach combines the benefits of transformer architecture with CNN-like inductive biases, making it particularly suitable for medical imaging tasks with limited training data.

---

## 3. Dataset Description

### 3.1 MSLD v2.0 (Monkeypox Skin Lesion Dataset Version 2.0)

The primary dataset used for training and evaluation is the Monkeypox Skin Lesion Dataset Version 2.0 (MSLD v2.0), which contains:

- **Total Images**: 755 images
- **Patients**: 541 patients
- **Classes**: 6 categories
- **Annotation**: Dermatologist-approved labels

### 3.2 Class Distribution

| Class | Image Count | Risk Level |
|-------|-------------|------------|
| Monkeypox | 284 | HIGH |
| Chickenpox | 75 | HIGH |
| Measles | 55 | HIGH |
| Cowpox | 66 | MEDIUM |
| HFMD | 161 | MEDIUM |
| Healthy | 114 | LOW |

### 3.3 Dataset Characteristics

The dataset presents several characteristics that influence the modeling approach:

1. **Class Imbalance**: The dataset exhibits significant class imbalance, with Monkeypox having approximately 5x more samples than Measles

2. **Multi-patient Distribution**: Images from 541 patients suggest potential need for patient-level splits to prevent data leakage

3. **Clinical Photography**: Images are collected from clinical settings with varying photography conditions

4. **Expert Verification**: All labels have been verified by dermatologists, ensuring high-quality ground truth

---

## 4. Methodology

### 4.1 Overview of Approach

The MpoxNet-V system employs a hybrid deep learning architecture that combines the strengths of Vision Transformers and Convolutional Neural Networks. The methodology encompasses:

1. **Data Preprocessing and Augmentation**: Comprehensive image preprocessing with medical-imaging-specific augmentations

2. **Dual-Branch Feature Extraction**: Parallel processing through two complementary branches

3. **Cross-Attention Gate Fusion**: Learned fusion mechanism for combining branch outputs

4. **Three-Phase Training Strategy**: Progressive unfreezing with curriculum-based learning

5. **Cross-Validation Evaluation**: Robust evaluation through k-fold cross-validation

### 4.2 Data Preprocessing

#### 4.2.1 Image Standardization

All input images undergo standardization to a consistent resolution of 224×224 pixels. This standardization ensures compatibility with pre-trained backbone models and enables efficient batch processing.

#### 4.2.2 Normalization

Image normalization follows the ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]) to leverage pre-trained weights effectively.

#### 4.2.3 Data Augmentation Strategy

A comprehensive augmentation pipeline addresses the limited dataset size and improves model generalization:

**Geometric Augmentations**:
- Random horizontal flip (probability: 0.5)
- Random vertical flip (probability: 0.5)
- Random rotation (±15 degrees)
- RandAugment (2 operations, magnitude 9)

**Color Augmentations**:
- Brightness adjustment (±0.2)
- Contrast adjustment (±0.15)
- Saturation adjustment (±0.15)
- Hue adjustment (±0.05)

**Cutout Augmentation**:
- Random erasing (probability: 0.25, scale: 0.02-0.12)

**Advanced Mix-Based Augmentations**:
- CutMix (probability: 0.3, beta parameter: 0.4)
- MixUp (probability: 0.2, alpha parameter: 0.2)

### 4.3 Model Architecture

#### 4.3.1 Dual-Branch Design Philosophy

The architecture employs a dual-branch design based on the principle of complementary feature extraction:

- **Branch 1 (Global Attention)**: Captures global spatial relationships and overall lesion structure through Vision Transformer architecture
- **Branch 2 (Local Texture)**: Extracts local texture patterns and fine-grained features through convolutional architecture

#### 4.3.2 Branch 1: DeiT-B (Data-efficient Image Transformer)

**Rationale**: 
- Provides global attention mechanisms for understanding overall lesion structure
- Distillation token enables transfer of CNN inductive biases
- Addresses the small-dataset limitation of standard Vision Transformers
- Pre-trained on ImageNet provides strong initialization

**Architecture Details**:
- Model variant: deit_base_distilled_patch16_224
- Output dimension: 1536-dimensional feature vector (768 × 2 from [CLS] and [DIST] tokens)
- Uses 12 transformer blocks with 12 attention heads

#### 4.3.3 Branch 2: EfficientNetB4

**Rationale**:
- EfficientNet architecture provides excellent trade-off between accuracy and computational efficiency
- Compound scaling captures multi-scale features
- Strong local texture representation complements transformer's global attention
- Pre-trained on ImageNet provides robust feature extraction

**Architecture Details**:
- Model variant: efficientnet_b4
- Output dimension: 1792-dimensional feature vector
- Uses inverted residual blocks with squeeze-excitation modules

#### 4.3.4 Cross-Attention Gate Fusion

The fusion mechanism employs a learned gating approach that dynamically weights the contributions of each branch:

**Design Principles**:
- Per-image dynamic weighting (not fixed weights)
- Learns to emphasize appropriate branch based on input characteristics
- Ensures α + β = 1 (normalized weights)

**Architecture**:
- Input: Concatenated features from both branches (2048 dimensions)
- Hidden layer: 256-dimensional with ReLU activation
- Dropout: 0.1 for regularization
- Output: 2-dimensional softmax for branch weights
- Fusion: α × Global_features + β × Local_features

**Interpretation**:
- α (alpha): Weight for DeiT-B branch (global attention features)
- β (beta): Weight for EfficientNetB4 branch (local texture features)

#### 4.3.5 Classification Head

The classification head processes the fused features:

- Input dimension: 1024 (fused feature dimension)
- Hidden layer: 512 dimensions with GELU activation
- Dropout: 0.35 for regularization
- Output: 6-dimensional logits for classification

### 4.4 Training Strategy

#### 4.4.1 Three-Phase Curriculum Learning

The training follows a progressive unfreezing approach with three distinct phases:

**Phase 1: Gate and Head Training (5 epochs)**
- Backbone models frozen (no gradient computation)
- Only Cross-Attention Gate and classification head trained
- Learning rate: 1e-3
- Purpose: Learn optimal fusion weights before backbone fine-tuning

**Phase 2: Partial Unfreeze (13 epochs)**
- Last 4 blocks of DeiT-B unfrozen
- Last 3 stages of EfficientNetB4 unfrozen
- Projection layers, gate, and head trained with higher learning rate
- Learning rate for backbones: 5e-5
- Learning rate for head components: 1e-4
- Early stopping patience: 12 epochs

**Phase 3: Full Fine-tuning (42 epochs)**
- All model parameters unfrozen
- Lower learning rate for gentle fine-tuning: 1e-4
- Learning rate schedule: Cosine annealing with warm restarts
- Early stopping patience: 12 epochs
- Purpose: Global optimum search with all parameters adjustable

#### 4.4.2 Loss Function

A combined loss function addresses class imbalance and focuses on hard examples:

**CombinedLoss = 0.7 × CrossEntropy + 0.3 × FocalLoss**

- **CrossEntropy with Class Weights**: Addresses dataset imbalance by weighting loss inversely to class frequency
- **Focal Loss (γ=2.0)**: Focuses training on hard-to-classify examples
- **Label Smoothing (0.1)**: Prevents overconfident predictions and improves generalization

#### 4.4.3 Optimization

- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate Scheduler**: CosineAnnealingWarmRestarts
  - Phase 2: T_0=6, T_mult=2
  - Phase 3: T_0=10, T_mult=2, η_min=1e-6
- **Gradient Clipping**: Maximum norm of 1.0 to prevent gradient explosion

#### 4.4.4 Evaluation Protocol

- **Cross-Validation**: 5-fold stratified cross-validation
- **Stratification**: Ensures equal class distribution across folds
- **Evaluation Metrics**:
  - Accuracy
  - F1-Score (macro-averaged)
  - Cohen's Kappa
  - ROC-AUC (macro-averaged, one-vs-rest)

---

## 5. Clinical Integration

### 5.1 Risk Stratification

The system provides clinical risk levels based on predicted disease:

| Disease | Risk Level | Clinical Action |
|---------|------------|-----------------|
| Monkeypox | HIGH | Isolate patient, seek immediate PCR confirmation |
| Chickenpox | HIGH | Consult dermatologist, avoid immunocompromised contact |
| Measles | HIGH | Isolate patient, report to public health authority |
| Cowpox | MEDIUM | Consult physician, usually self-limiting |
| HFMD | MEDIUM | Rest and hydration, monitor for complications |
| Healthy | LOW | Routine follow-up if symptoms persist |

### 5.2 Attention Visualization

The system provides interpretable attention maps through DeiT-B attention rollout, enabling clinicians to understand which image regions influenced the prediction. This transparency supports clinical trust and facilitates expert review of predictions.

---

## 6. System Architecture and Deployment

### 6.1 API Design

The system exposes a RESTful API built with FastAPI, providing:

**Prediction Endpoints**:
- Single image classification: POST /predict
- Batch classification (up to 20 images): POST /predict/batch
- Attention visualization: POST /predict/attention
- Class information: GET /predict/classes

**Training Endpoints**:
- Start training: POST /train/start
- Training status: GET /train/status
- Training results: GET /train/results
- Load model: POST /train/load

**System Endpoints**:
- Health check: GET /health

### 6.2 Web Interface

A user-friendly web interface enables:
- Drag-and-drop image upload
- Real-time classification
- Probability visualization
- Attention overlay display
- Risk level display with clinical recommendations

### 6.3 Deployment Options

The system supports multiple deployment scenarios:

1. **Local Development**: Direct execution via uvicorn
2. **Container Deployment**: Docker containerization for reproducibility
3. **Cloud Deployment**: Scalable architecture suitable for cloud platforms

---

## 7. Evaluation Framework

### 7.1 Primary Metrics

The evaluation framework employs multiple metrics to comprehensively assess model performance:

1. **Accuracy**: Overall proportion of correct predictions
2. **Macro F1-Score**: Harmonic mean of precision and recall, averaged across classes
3. **Cohen's Kappa**: Agreement measure accounting for chance
4. **ROC-AUC**: Area under the receiver operating characteristic curve

### 7.2 Secondary Metrics

1. **Per-class Precision and Recall**: Detailed performance breakdown
2. **Confusion Matrix**: Class-level error analysis
3. **Gate Weight Distribution**: Analysis of branch contribution patterns

### 7.3 Statistical Analysis

- Cross-validation provides mean and standard deviation for all metrics
- Enables assessment of model stability across different data splits
- Identifies potential overfitting through variance analysis

---

## 8. Implementation Methodology Summary

### 8.1 Development Approach

The implementation follows a systematic methodology:

1. **Requirements Analysis**: Understanding clinical needs and technical constraints
2. **Architecture Design**: Novel dual-branch transformer-CNN fusion
3. **Preprocessing Pipeline**: Medical-imaging-specific augmentation
4. **Training Implementation**: Three-phase curriculum learning
5. **API Development**: RESTful service with comprehensive endpoints
6. **Validation**: 5-fold cross-validation with multiple metrics

### 8.2 Key Innovations

1. **Cross-Attention Gate**: Dynamic per-image fusion weights
2. **DeiT-B Integration**: Vision transformer with distillation for small datasets
3. **Combined Loss**: Focal loss + weighted cross-entropy for imbalanced data
4. **Clinical Risk Mapping**: Actionable output beyond raw predictions

### 8.3 Reproducibility Considerations

- Fixed random seeds for reproducible experiments
- Comprehensive hyperparameter documentation
- Docker containerization for environment consistency
- Public dataset (MSLD v2.0) for independent validation

---

## 9. Expected Outcomes and Contributions

### 9.1 Performance Targets

The primary performance target is achieving classification accuracy exceeding the current SwinTransformer baseline of 93.71% on the MSLD v2.0 dataset.

### 9.2 Research Contributions

1. **Novel Architecture**: Dual-branch Vision Transformer with Cross-Attention Gate fusion
2. **Medical Imaging Application**: Demonstrates effectiveness of hybrid transformer-CNN architectures
3. **Clinical Integration**: Risk stratification and recommendation system
4. **Interpretability**: Attention visualization for clinical trust

### 9.3 Practical Contributions

1. **Open-Source Implementation**: Complete codebase for reproduction
2. **API Service**: Deployable classification service
3. **Web Interface**: User-friendly demonstration tool
4. **Documentation**: Comprehensive technical documentation

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. **Dataset Size**: 755 images is relatively small for deep learning
2. **Class Imbalance**: Significant variation in class distribution
3. **Single Dataset**: Performance on external datasets not yet validated
4. **2D Images**: Does not utilize 3D or multi-view imaging

### 10.2 Future Research Directions

1. **External Validation**: Test on additional datasets from different sources
2. **Multi-modal Integration**: Incorporate patient metadata and clinical notes
3. **Ensemble Methods**: Combine multiple model architectures
4. **Temporal Analysis**: Track lesion progression over time
5. **Clinical Trial**: Prospective validation in clinical settings

---

## 11. Conclusion

MpoxNet-V represents a significant step toward automated monkeypox skin lesion classification. By combining the global attention capabilities of Vision Transformers with the local texture extraction of EfficientNet, through a learned Cross-Attention Gate fusion mechanism, the system achieves a balance between accuracy and interpretability. The three-phase training curriculum, combined loss function, and comprehensive data augmentation address the challenges of limited medical imaging datasets.

The system's clinical risk stratification and attention visualization provide meaningful outputs for healthcare providers, supporting rather than replacing clinical expertise. With the goal of exceeding 93.71% accuracy, this research contributes both to the advancement of medical image analysis and to the global response to monkeypox and similar infectious diseases.

---

## References

1. Uysal et al. (2023). CNN-LSTM approach to monkeypox classification. MDPI Diagnostics, 87% accuracy, Kappa=0.8222.

2. Jaradat et al. (2023). MobileNetV2 for monkeypox detection. PMC10001976, 98.16% accuracy.

3. Vega et al. (2023). Dataset quality critique of MSLD. PMC10010024.

4. Al-Hammuri et al. (2023). Vision Transformers in digital health survey.

5. Dosovitskiy et al. (2020). Original Vision Transformer (ViT) paper.

6. Touvron et al. (2021). Data-efficient Image Transformers (DeiT) with distillation.

7. Vuran et al. (2025). SwinTransformer achieving 93.71% on MSLD v2.0 (current SOTA).

---

## Appendix: Technical Specifications Summary

- **Input Resolution**: 224 × 224 pixels
- **Model Parameters**: Approximately 107 million
- **Number of Classes**: 6
- **Training Protocol**: 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, F1-Score, Cohen's Kappa, ROC-AUC
- **Deployment**: FastAPI REST API with Docker support
- **Dataset**: MSLD v2.0 (755 images, 541 patients)

---

*Document Version: 1.0*
*Date: 2024*
*Project: MpoxNet-V Research Implementation*
