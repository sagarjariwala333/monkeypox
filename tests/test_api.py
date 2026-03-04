"""
tests/test_api.py — API endpoint tests
Run: pytest tests/test_api.py -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import sys
sys.path.insert(0, "..")

from app.main import app

client = TestClient(app)


def make_fake_image(size=(224, 224), mode="RGB") -> bytes:
    """Generate a synthetic image for testing."""
    img = Image.new(mode, size, color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─── Health ────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "classes"      in data
    assert len(data["classes"]) == 6


# ─── Classes ───────────────────────────────────────────────────────────────────

def test_get_classes():
    r = client.get("/predict/classes")
    assert r.status_code == 200
    data = r.json()
    assert "classes"      in data
    assert len(data["classes"]) == 6
    class_names = [c["name"] for c in data["classes"]]
    assert "Monkeypox"  in class_names
    assert "Chickenpox" in class_names
    assert "Healthy"    in class_names


# ─── Prediction ────────────────────────────────────────────────────────────────

def test_predict_valid_image():
    img_bytes = make_fake_image()
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "prediction"          in data
    assert "confidence"          in data
    assert "confidence_pct"      in data
    assert "all_probabilities"   in data
    assert "clinical_risk"       in data
    assert "recommendation"      in data
    assert len(data["all_probabilities"]) == 6
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["clinical_risk"] in ("HIGH", "MEDIUM", "LOW")


def test_predict_invalid_file_type():
    r = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_predict_png():
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200


def test_predict_batch():
    imgs = [make_fake_image() for _ in range(3)]
    files = [
        ("files", (f"img{i}.jpg", img, "image/jpeg"))
        for i, img in enumerate(imgs)
    ]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["total_images"] == 3
    assert len(data["results"])  == 3
    assert "summary"             in data


def test_predict_batch_too_many():
    imgs  = [make_fake_image() for _ in range(21)]
    files = [("files", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(imgs)]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 400


# ─── Training ──────────────────────────────────────────────────────────────────

def test_training_status_idle():
    r = client.get("/train/status")
    assert r.status_code == 200
    data = r.json()
    assert "status"        in data
    assert "current_phase" in data
    assert "current_fold"  in data


def test_training_results_not_done():
    r = client.get("/train/results")
    # Should 400 if not completed
    assert r.status_code in (200, 400)


def test_load_nonexistent_model():
    r = client.post("/train/load?model_path=./does_not_exist.pt&num_classes=6")
    assert r.status_code == 404


def test_saved_models_list():
    r = client.get("/train/saved")
    assert r.status_code == 200
    assert "saved_models" in r.json()
