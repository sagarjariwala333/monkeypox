import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report
from utils.dataset import get_val_transform
from models.mpoxnet_v import MpoxNetV
import os
import sys

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = "./archive/Original Images/Original Images"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        sys.exit(1)

    dataset = ImageFolder(root=data_dir, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # We have 2 classes: Monkey Pox, Others
    model = MpoxNetV(num_classes=2).to(device)

    best_model = "saved_models/mpoxnet_v_fold1.pt"
    if os.path.exists(best_model):
        print(f"Loading weights from {best_model}...")
        ckpt = torch.load(best_model, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Warning: Previous testing was aborted before the final fold completed saving.")
        print("Evaluating with pre-trained ImageNet baseline weights (untrained on your dataset).")

    model.eval()
    preds = []
    targets = []

    print("Evaluating dataset...")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            outputs = model(imgs.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.numpy())
            print(f"Processed batch {i+1}/{len(loader)}")

    acc = accuracy_score(targets, preds)
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=dataset.classes))

if __name__ == "__main__":
    evaluate()
