"""Flask API exposing the trained RoadAI U-Net to the React frontend."""
from __future__ import annotations

import base64
import sys
import traceback
import uuid
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_PATH as CONFIG_MODEL_PATH, POSTPROCESS_MIN_COMPONENT, THRESHOLD  # noqa: E402
from dataset import load_image_for_prediction  # noqa: E402
from utils import clean_mask  # noqa: E402

# Defaults to models/best_model.keras. Set ROADAI_MODEL to opt into a validated candidate.
MODEL_PATH = CONFIG_MODEL_PATH
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _load_model_once() -> tf.keras.Model:
    """Load the model once and print actionable startup diagnostics on failure."""
    print(f"Backend directory: {BACKEND_DIR}", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    print(f"Uploads directory: {UPLOAD_DIR}", flush=True)
    print(f"Outputs directory: {OUTPUT_DIR}", flush=True)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model is missing: {MODEL_PATH}")
    print("Loading model...", flush=True)
    try:
        # `compile=False` avoids deserializing training-only custom losses/metrics.
        model = load_model(MODEL_PATH, compile=False)
    except Exception:
        print("Model loading failed. Full exception follows:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    print("Model loaded successfully.", flush=True)
    return model


# Loaded exactly once when `python backend/app.py` starts.
MODEL = _load_model_once()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
# Vite uses 5173 by default but moves to 5174+ when that port is occupied.
CORS(app, resources={r"/*": {"origins": r"^http://(localhost|127\.0\.0\.1):\d+$"}})


@app.get("/")
def index():
    """A simple browser-visible status page; the React UI remains on port 5173."""
    return jsonify(success=True, message="RoadAI backend is running. Use GET /health or POST /predict."), 200


def _as_data_url(image: np.ndarray, grayscale: bool = False) -> str:
    """Encode a NumPy image as a browser-ready base64 PNG URL."""
    if grayscale:
        encoded_image = image
    else:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", encoded_image)
    if not success:
        raise ValueError("Could not encode prediction image as PNG")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _save_outputs(request_id: str, original: np.ndarray, mask: np.ndarray, overlay: np.ndarray) -> None:
    """Persist recent inference images for backend-side inspection."""
    cv2.imwrite(str(OUTPUT_DIR / f"{request_id}_original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / f"{request_id}_mask.png"), mask)
    cv2.imwrite(str(OUTPUT_DIR / f"{request_id}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _predict_image(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the exact training-time preprocessing and U-Net inference for one file."""
    image = load_image_for_prediction(path)
    probabilities = MODEL.predict(image[np.newaxis, ...], verbose=0)[0]
    binary_mask = clean_mask(probabilities, THRESHOLD, POSTPROCESS_MIN_COMPONENT).astype(np.uint8) * 255
    original = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    overlay = original.copy()
    overlay[binary_mask > 0] = (255, 64, 40)
    return original, binary_mask, overlay, probabilities


def _analysis_difference(before_mask: np.ndarray, after_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Colour-code road continuity: green unchanged, red lost, yellow changed/new."""
    before = before_mask > 0
    after = after_mask > 0
    unchanged = before & after
    lost = before & ~after
    changed = ~before & after
    difference = np.zeros((*before.shape, 3), dtype=np.uint8)
    difference[unchanged] = (34, 197, 94)   # green: road is still detected
    difference[lost] = (239, 68, 68)        # red: road disappeared after the event
    difference[changed] = (250, 204, 21)    # yellow: changed/new road region
    before_pixels = int(before.sum())
    after_pixels = int(after.sum())
    lost_pixels = int(lost.sum())
    metres_per_pixel = 0.5  # Same clearly labelled estimate used by the prediction UI.
    return difference, {
        "road_pixels_before": before_pixels,
        "road_pixels_after": after_pixels,
        "road_pixels_lost": lost_pixels,
        "damage_percentage": round((lost_pixels / before_pixels * 100) if before_pixels else 0.0, 2),
        "estimated_road_length_before": round(before_pixels * metres_per_pixel, 2),
        "estimated_road_length_after": round(after_pixels * metres_per_pixel, 2),
        "estimated_length_lost": round(lost_pixels * metres_per_pixel, 2),
    }


def _uploaded_file(field_name: str, request_id: str) -> Path:
    """Validate and persist one multipart upload for the duration of a request."""
    if field_name not in request.files:
        raise ValueError(f"Missing multipart field: {field_name}")
    upload = request.files[field_name]
    if not upload.filename:
        raise ValueError(f"No {field_name.replace('_', ' ')} was selected")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported formats: TIFF, GeoTIFF, PNG, JPG")
    path = UPLOAD_DIR / f"{request_id}_{field_name}_{secure_filename(upload.filename)}"
    upload.save(path)
    return path


@app.post("/predict")
def predict():
    """Accept an image as multipart form data and return U-Net visualizations."""
    if "image" not in request.files:
        return jsonify(success=False, error="Missing multipart field: image"), 400
    upload = request.files["image"]
    if not upload.filename:
        return jsonify(success=False, error="No image was selected"), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, error="Supported formats: TIFF, GeoTIFF, PNG, JPG"), 400

    request_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{request_id}_{secure_filename(upload.filename)}"
    try:
        upload.save(upload_path)
        original, binary_mask, overlay, probabilities = _predict_image(upload_path)
        _save_outputs(request_id, original, binary_mask, overlay)
        return jsonify(
            success=True,
            original=_as_data_url(original),
            mask=_as_data_url(binary_mask, grayscale=True),
            overlay=_as_data_url(overlay),
            confidence=round(float(probabilities.mean()), 4),
        )
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify(success=False, error=f"Prediction failed: {error}"), 500
    finally:
        upload_path.unlink(missing_ok=True)


@app.post("/disaster-analysis")
def disaster_analysis():
    """Compare U-Net road predictions from before and after a disaster event."""
    request_id = uuid.uuid4().hex
    paths: list[Path] = []
    started = perf_counter()
    try:
        before_path = _uploaded_file("before_image", request_id)
        paths.append(before_path)
        after_path = _uploaded_file("after_image", request_id)
        paths.append(after_path)
        before_original, before_mask, before_overlay, before_probabilities = _predict_image(before_path)
        after_original, after_mask, after_overlay, after_probabilities = _predict_image(after_path)
        difference, statistics = _analysis_difference(before_mask, after_mask)
        statistics["prediction_time"] = round(perf_counter() - started, 3)
        statistics["model"] = "U-Net Fine Tuned"
        statistics["threshold"] = THRESHOLD
        _save_outputs(f"{request_id}_before", before_original, before_mask, before_overlay)
        _save_outputs(f"{request_id}_after", after_original, after_mask, after_overlay)
        cv2.imwrite(str(OUTPUT_DIR / f"{request_id}_difference.png"), cv2.cvtColor(difference, cv2.COLOR_RGB2BGR))
        return jsonify(
            success=True,
            before_image=_as_data_url(before_original), before_mask=_as_data_url(before_mask, grayscale=True),
            before_overlay=_as_data_url(before_overlay), after_image=_as_data_url(after_original),
            after_mask=_as_data_url(after_mask, grayscale=True), after_overlay=_as_data_url(after_overlay),
            difference_map=_as_data_url(difference), statistics=statistics,
            before_confidence=round(float(before_probabilities.mean()), 4),
            after_confidence=round(float(after_probabilities.mean()), 4),
        )
    except ValueError as error:
        return jsonify(success=False, error=str(error)), 400
    except Exception as error:
        app.logger.exception("Disaster analysis failed")
        return jsonify(success=False, error=f"Disaster analysis failed: {error}"), 500
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify(success=True, model_loaded=True)


if __name__ == "__main__":
    print("Backend started.", flush=True)
    print("Listening on port 5000.", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
