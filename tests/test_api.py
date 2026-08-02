import io
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # type: ignore
from PIL import Image  # type: ignore

from app.main import app

client = TestClient(app)


def test_health_check_endpoint() -> None:
    """Verifies /api/health endpoint returns 200 OK and valid status dict."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint_valid_image() -> None:
    """Verifies /api/predict endpoint with a generated RGB leaf test image."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    response = client.post("/api/predict?xai_method=all", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "class_name" in data
    assert "disease_title" in data
    assert "confidence" in data
    assert "gradcam_heatmap_b64" in data
    assert data["ig_heatmap_b64"] is not None
    assert data["shap_heatmap_b64"] is not None
    assert data["gradcam_heatmap_b64"].startswith("data:image/png;base64,")


if __name__ == "__main__":
    print("Running test_health_check_endpoint...")
    test_health_check_endpoint()
    print("Running test_predict_endpoint_valid_image...")
    test_predict_endpoint_valid_image()
    print("All API integration tests passed successfully!")

