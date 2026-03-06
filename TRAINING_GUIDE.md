# Training and Testing MpoxNet-V on Custom Datasets

This document provides a comprehensive guide on how the MpoxNet-V Vision Transformer is trained and evaluated using custom datasets like the `archive` Monkey Pox dataset.

## 1. Dataset Preparation

The pipeline expects data to be organized in a standard PyTorch `ImageFolder` structure. This means the root directory should contain subdirectories, where each subdirectory corresponds to a class label, and contains the images for that class.

### Dataset Size and Composition
The `archive` dataset used for this project has the following composition:

*   **Total Images**: 228
*   **Classes**: 2
    *   `Monkey Pox`: 102 images
    *   `Others`: 126 images

### Training Duration and Complexity
Training the MpoxNet-V model is a computationally intensive process. Several factors contribute to the total training time:

1.  **Dual-Branch Architecture**: The model processes each image through two massive backbones simultaneously:
    *   **DeiT-B (Vision Transformer)**: ~86 Million parameters for global feature extraction.
    *   **EfficientNet-B4 (CNN)**: ~19 Million parameters for local texture analysis.
    The total parameter count is approximately **107 million**, and the model employs a **Cross-Attention Gate** for dynamic fusion.

2.  **Three-Phase Curriculum Learning**: Each fold undergoes three distinct training phases with progressive unfreezing (Gate -> Partial Unfreeze -> Full Fine-tuning), totaling **60 epochs per fold**.

3.  **5-Fold Cross-Validation**: To ensure clinical robustness and reliability, the model is trained from scratch 5 times on different data splits. This results in a total of **300 training epochs** for a full run.

**Example Structure:**
```
archive/Original Images/Original Images/
├── Monkey Pox/ (102 images)
│   ├── M01_01.jpg
│   ├── M01_02.jpg
│   └── ...
└── Others/ (126 images)
    ├── NM01_01.jpg
    ├── NM01_02.jpg
    └── ...
```

This structure automatically defines the classes: `["Monkey Pox", "Others"]`.

## 2. Configuring the Training Run

Training is initiated via the `/train/start` REST API endpoint. You need to create a JSON file (e.g., `full_train_config.json`) defining the hyperparameters and pointing to the dataset directory.

**Example Configuration (`full_train_config.json`):**
```json
{
  "data_dir": "./archive/Original Images/Original Images",
  "num_folds": 5,
  "batch_size": 16,
  "num_classes": 2,
  "phase1_epochs": 5,
  "phase2_epochs": 13,
  "phase3_epochs": 42
}
```

*   `data_dir`: The relative or absolute path to the root of your prepared dataset.
*   `num_folds`: The number of folds for cross-validation (e.g., 5).
*   `num_classes`: The number of subdirectories (classes) in your data directory.
*   `phaseX_epochs`: The number of epochs to train for each of the three progressive curriculum phases.

## 3. Initiating and Monitoring Training

### Starting the Training
With the FastAPI server running (`python -m uvicorn app.main:app`), start the background training process by sending a POST request with your configuration file.

**Using PowerShell:**
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/train/start -Method POST -ContentType "application/json" -InFile full_train_config.json
```

**Using cURL:**
```bash
curl -X POST http://127.0.0.1:8000/train/start -H "Content-Type: application/json" -d @full_train_config.json
```

### Monitoring the Status
The training runs as a background thread pool task so it doesn't block the API. You can poll the live status at any time by visiting or querying:

`GET http://127.0.0.1:8000/train/status`

This endpoints returns a JSON object summarizing the active fold, phase, epoch, loss, and F1 validation score.

### Saving the Weights
For each fold, upon completion of Phase 3, the optimal model weights are automatically saved to the `saved_models/` directory in the project root (e.g., `saved_models/mpoxnet_v_fold1.pt`).

## 4. Evaluating the Model

Once a fold finishes training and the `.pt` file is saved, you can test the model's accuracy on the entire dataset to generate a detailed Classification Report (precision, recall, f1-score).

A standalone script `calculate_accuracy.py` has been created for this purpose.

**How to run it:**
```bash
python calculate_accuracy.py
```

**What the script does:**
1. Loads the latest optimal weights from `saved_models/mpoxnet_v_fold1.pt`.
2. Instantiates the MpoxNet-V model configured for `num_classes=2`.
3. Passes every image in the dataset through the model using `torch.no_grad()`.
4. Compares predictions with the true folder labels.
5. Prints the overall accuracy percentage and the full `sklearn` classification report.

## 5. Single Image Prediction

To predict a single new image against the running API (which automatically loads the best saved model state on startup):

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/test_image.jpg"
```

The response provides the predicted class, confidence, comprehensive probability spread across all classes, and the dynamic `Gate Alpha/Beta` weights used by the model for that image.
