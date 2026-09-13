"""Generate road overlays for every image in the configured test split."""
import tensorflow as tf
from config import MODEL_PATH, RESULTS_DIR
from predict import predict_image
from preprocess import load_splits

def main():
    if not MODEL_PATH.is_file(): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False); images, _ = load_splits()["test"]
    for image in images: predict_image(model, image, RESULTS_DIR / "predictions")
    print(f"Saved {len(images)} test predictions to {RESULTS_DIR / 'predictions'}")

if __name__ == "__main__": main()
