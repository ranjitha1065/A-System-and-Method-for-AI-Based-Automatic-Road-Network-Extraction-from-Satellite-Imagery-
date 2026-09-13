"""Segmentation losses, metrics, and reusable result visualizations."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import THRESHOLD

@tf.keras.utils.register_keras_serializable(package="road")
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3); intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    return tf.reduce_mean((2 * intersection + smooth) / (tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + smooth))

@tf.keras.utils.register_keras_serializable(package="road")
def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)); union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=(1, 2, 3))
    return tf.reduce_mean((intersection + smooth) / (union + smooth))

@tf.keras.utils.register_keras_serializable(package="road")
def dice_loss(y_true, y_pred): return 1.0 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package="road")
def bce_dice_loss(y_true, y_pred):
    """Balanced, stable loss for sparse road segmentation."""
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)

def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=bce_dice_loss, metrics=[dice_coef, iou_score, tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
    return model

def plot_history(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = {"loss": "loss_curve.png", "dice_coef": "dice_curve.png", "iou_score": "iou_curve.png", "precision": "precision_curve.png", "recall": "recall_curve.png"}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (metric, filename) in zip(axes.flat, metrics.items()):
        axis.plot(history[metric], label="Train")
        axis.plot(history[f"val_{metric}"], label="Validation")
        axis.set(title=metric.replace("_", " ").title(), xlabel="Epoch"); axis.legend(); axis.figure.tight_layout()
        axis.figure.savefig(output / "training_history.png", dpi=150)
        single, single_axis = plt.subplots(); single_axis.plot(history[metric], label="Train"); single_axis.plot(history[f"val_{metric}"], label="Validation"); single_axis.legend(); single_axis.set(title=metric.replace("_", " ").title(), xlabel="Epoch"); single.tight_layout(); single.savefig(output / filename, dpi=150); plt.close(single)
    axes.flat[-1].axis("off"); figure.tight_layout(); figure.savefig(output / "training_history.png", dpi=150); plt.close(figure)

def prediction_images(image: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = (np.clip(image, 0, 1) * 255).astype(np.uint8); mask = (probability.squeeze() >= THRESHOLD).astype(np.uint8) * 255
    overlay = original.copy(); overlay[mask > 0] = (255, 64, 40)
    return original, mask, overlay

def clean_mask(probability: np.ndarray, threshold: float = THRESHOLD, min_component: int = 0) -> np.ndarray:
    """Optional conservative OpenCV mask cleanup; disabled by default (min_component=0)."""
    import cv2
    mask = (np.squeeze(probability) >= threshold).astype(np.uint8)
    if min_component <= 0:
        return mask
    # A tiny closing reconnects one-pixel gaps without changing road geometry broadly.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component:
            cleaned[labels == label] = 1
    return cleaned

def save_prediction_images(image: np.ndarray, probability: np.ndarray, output_dir: str | Path, stem: str) -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); original, mask, overlay = prediction_images(image, probability)
    import cv2
    cv2.imwrite(str(output / f"{stem}_original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR)); cv2.imwrite(str(output / f"{stem}_mask.png"), mask); cv2.imwrite(str(output / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
