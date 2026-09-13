"""Leakage-safe evaluation, threshold scoring, and current-vs-candidate comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from config import CURRENT_MODEL_PATH, IMPROVED_MODEL_PATH, MODEL_PATH, RESULTS_DIR, THRESHOLD, THRESHOLD_CANDIDATES
from dataset import load_image_for_prediction, load_mask_for_prediction, make_dataset
from preprocess import load_splits

def collect_predictions(model: tf.keras.Model, data):
    truth, probability = [], []
    for image_batch, mask_batch in data:
        truth.append(mask_batch.numpy().ravel()); probability.append(model(image_batch, training=False).numpy().ravel())
    return np.concatenate(truth).astype(np.uint8), np.concatenate(probability).astype(np.float32)

def metrics_at_threshold(truth, probability, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.uint8)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, average="binary", zero_division=0)
    intersection = np.logical_and(truth, prediction).sum(); union = np.logical_or(truth, prediction).sum()
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(truth, prediction)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "dice": float(2 * intersection / (truth.sum() + prediction.sum()) if truth.sum() + prediction.sum() else 1.0), "iou": float(intersection / union if union else 1.0)}

def score_thresholds(truth, probability, thresholds=THRESHOLD_CANDIDATES): return [metrics_at_threshold(truth, probability, threshold) for threshold in thresholds]

def threshold_from_sidecar(path: Path) -> float:
    try: return float(json.loads(path.with_suffix(".threshold.json").read_text())["threshold"])
    except (OSError, ValueError, KeyError): return THRESHOLD

def save_plots(truth, probability, threshold, output: Path):
    prediction = (probability >= threshold).astype(np.uint8)
    ConfusionMatrixDisplay(confusion_matrix(truth, prediction, labels=[0, 1]), display_labels=["Background", "Road"]).plot(cmap="Blues", values_format="d")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=150); plt.close()
    if len(np.unique(truth)) == 2:
        fpr, tpr, _ = roc_curve(truth, probability); plt.plot(fpr, tpr, label=f"AUC {roc_auc_score(truth, probability):.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(output / "roc_curve.png", dpi=150); plt.close()

def evaluate_model(path: Path, split: str, output_name: str) -> dict[str, float]:
    splits = load_splits(); images, masks = splits[split]
    if not masks: raise ValueError(f"The configured {split} split has no labels, so it cannot be evaluated.")
    model = tf.keras.models.load_model(path, compile=False); truth, probability = collect_predictions(model, make_dataset(images, masks))
    threshold = threshold_from_sidecar(path); report = metrics_at_threshold(truth, probability, threshold)
    report.update({"model": str(path), "split": split, "sample_count": len(images)})
    RESULTS_DIR.mkdir(exist_ok=True); (RESULTS_DIR / output_name).write_text(json.dumps(report, indent=2)); save_plots(truth, probability, threshold, RESULTS_DIR)
    return report

def compare_models(split: str = "test"):
    """Evaluate both checkpoints on the same labelled split; no tuning occurs here."""
    if not IMPROVED_MODEL_PATH.is_file(): raise FileNotFoundError(f"Candidate model missing: {IMPROVED_MODEL_PATH}")
    current = evaluate_model(CURRENT_MODEL_PATH, split, "evaluation_current.json"); improved = evaluate_model(IMPROVED_MODEL_PATH, split, "evaluation_improved.json")
    rows = ["Metric,Current Model,Improved Model,Change"]
    for metric in ("dice", "iou", "precision", "recall", "accuracy"): rows.append(f"{metric},{current[metric]:.6f},{improved[metric]:.6f},{improved[metric] - current[metric]:+.6f}")
    (RESULTS_DIR / "model_comparison.csv").write_text("\n".join(rows) + "\n")
    images, masks = load_splits()[split]
    visual_dir = RESULTS_DIR / "model_comparison_visuals"; visual_dir.mkdir(exist_ok=True)
    current_model = tf.keras.models.load_model(CURRENT_MODEL_PATH, compile=False); improved_model = tf.keras.models.load_model(IMPROVED_MODEL_PATH, compile=False)
    for index, (image_path, mask_path) in enumerate(zip(images[:5], masks[:5])):
        image = load_image_for_prediction(image_path); truth = load_mask_for_prediction(mask_path).squeeze()
        old = current_model(image[None, ...], training=False).numpy().squeeze() >= threshold_from_sidecar(CURRENT_MODEL_PATH)
        new = improved_model(image[None, ...], training=False).numpy().squeeze() >= threshold_from_sidecar(IMPROVED_MODEL_PATH)
        figure, axes = plt.subplots(1, 4, figsize=(14, 4))
        for axis, data, title, cmap in zip(axes, (image, truth, old, new), ("Original", "Ground Truth", "Current Model", "Improved Model"), (None, "gray", "gray", "gray")):
            axis.imshow(data, cmap=cmap); axis.set_title(title); axis.axis("off")
        figure.tight_layout(); figure.savefig(visual_dir / f"comparison_{index + 1}.png", dpi=150); plt.close(figure)
    return current, improved

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=MODEL_PATH); parser.add_argument("--split", default="test", choices=("validation", "test")); parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        current, improved = compare_models(args.split); print(json.dumps({"current": current, "improved": improved}, indent=2))
    else: print(json.dumps(evaluate_model(args.model, args.split, "evaluation_report.json"), indent=2))

if __name__ == "__main__": main()
