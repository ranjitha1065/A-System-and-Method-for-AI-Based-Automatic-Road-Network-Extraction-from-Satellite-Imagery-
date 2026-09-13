"""Run a single road prediction; supports any TIFF/GeoTIFF/JPG/PNG image."""
import argparse
from pathlib import Path
import tensorflow as tf
from config import MODEL_PATH, RESULTS_DIR
from dataset import load_image_for_prediction
from utils import save_prediction_images

def predict_image(model, image_path: str | Path, output_dir=RESULTS_DIR / "predictions"):
    image_path = Path(image_path); image = load_image_for_prediction(image_path)
    probability = model.predict(image[None, ...], verbose=0)[0]; save_prediction_images(image, probability, output_dir, image_path.stem)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("image", help="Image path to segment"); args = parser.parse_args()
    if not MODEL_PATH.is_file(): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    predict_image(tf.keras.models.load_model(MODEL_PATH, compile=False), args.image)
    print(f"Saved prediction images to {RESULTS_DIR / 'predictions'}")

if __name__ == "__main__": main()
