import io
from PIL import Image
from pathlib import Path

def create_dummy_images():
    base_path = Path("data")
    colors = {
        "Monkeypox": (120, 80, 60),  # Brownish
        "Healthy": (200, 220, 200),  # Light Greenish
        "Chickenpox": (220, 180, 180), # Reddish
        "Measles": (255, 100, 100),    # Bright Red
        "Cowpox": (150, 150, 100),     # Dull Yellow
        "HFMD": (200, 100, 200)        # Purple
    }
    
    for cls, color in colors.items():
        cls_dir = base_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=color)
            img_path = cls_dir / f"dummy_{i}.jpg"
            img.save(img_path, format="JPEG")
            print(f"Created {img_path} with color {color}")

if __name__ == "__main__":
    create_dummy_images()
