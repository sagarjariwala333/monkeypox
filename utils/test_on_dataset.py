import os
import requests
import argparse
from pathlib import Path

def test_on_dataset(data_dir, api_url="http://localhost:8000"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    results = []
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found classes: {classes}")
    
    for cls in classes:
        cls_dir = data_path / cls
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        
        print(f"Testing {len(images)} images in class: {cls}")
        
        for img_path in images:
            with open(img_path, "rb") as f:
                try:
                    r = requests.post(f"{api_url}/predict", files={"file": (img_path.name, f, "image/jpeg")})
                    if r.status_code == 200:
                        prediction = r.json()
                        results.append({
                            "file": img_path.name,
                            "actual": cls,
                            "predicted": prediction["prediction"],
                            "confidence": prediction["confidence_pct"]
                        })
                    else:
                        print(f"Error predicting {img_path.name}: {r.status_code}")
                except Exception as e:
                    print(f"Failed to connect to API: {e}")
                    return

    # Simple Accuracy Calculation
    correct = sum(1 for res in results if res["actual"].lower() == res["predicted"].lower())
    total = len(results)
    if total > 0:
        print(f"\nTest Summary:")
        print(f"Total Images: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Overall Accuracy: {(correct/total)*100:.2f}%")
    else:
        print("No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MpoxNet-V on a local dataset.")
    parser.add_argument("data_dir", type=str, help="Path to the dataset (e.g., ./data/MSLD_v2)")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", help="FastAPI server URL")
    
    args = parser.parse_args()
    test_on_dataset(args.data_dir, args.api_url)
