"""Fine-tune a candidate model without ever overwriting the production checkpoint."""
import json
import random
import numpy as np
import tensorflow as tf
from config import EPOCHS, FINE_TUNE_LEARNING_RATE, FREEZE_ENCODER_EPOCHS, IMPROVED_MODEL_PATH, LEARNING_RATE, MODEL_DIR, CURRENT_MODEL_PATH, RANDOM_SEED, RESULTS_DIR, THRESHOLD_CANDIDATES
from dataset import inspect_masks, make_dataset
from preprocess import load_splits, validate_splits
from unet import freeze_encoder, unfreeze_all
from utils import compile_model, plot_history

def _threshold_selection(model, validation_data):
    from evaluate import collect_predictions, metrics_at_threshold, score_thresholds
    from utils import clean_mask
    truth, probability = collect_predictions(model, validation_data)
    table = score_thresholds(truth, probability, THRESHOLD_CANDIDATES)
    best = max(table, key=lambda row: (row["dice"], row["iou"], row["precision"]))
    postprocess = []
    for minimum in (0, 4, 8, 16):
        cleaned = np.concatenate([clean_mask(item, best["threshold"], minimum).ravel() for item in probability.reshape((-1, 256, 256))])
        postprocess.append({"min_component": minimum, **metrics_at_threshold(truth, cleaned, .5)})
    winning_clean = max(postprocess, key=lambda row: (row["dice"], row["iou"]))
    selected_minimum = winning_clean["min_component"] if winning_clean["dice"] > best["dice"] else 0
    best["postprocess_min_component"] = selected_minimum
    (RESULTS_DIR / "threshold_selection.json").write_text(json.dumps({"selected": best, "all_thresholds": table, "postprocess_comparison": postprocess}, indent=2))
    IMPROVED_MODEL_PATH.with_suffix(".threshold.json").write_text(json.dumps(best, indent=2))
    print(f"Selected validation threshold: {best['threshold']:.2f} (Dice {best['dice']:.4f}, IoU {best['iou']:.4f}); postprocess minimum={selected_minimum}")
    return best

def _callbacks():
    return [tf.keras.callbacks.ModelCheckpoint(IMPROVED_MODEL_PATH, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1), tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=4, min_lr=1e-7, verbose=1), tf.keras.callbacks.CSVLogger(RESULTS_DIR / "training_history_improved.csv")]

def _extend(target, history):
    for key, values in history.history.items(): target.setdefault(key, []).extend(values)

def main():
    random.seed(RANDOM_SEED); tf.keras.utils.set_random_seed(RANDOM_SEED); MODEL_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    if not CURRENT_MODEL_PATH.is_file(): raise FileNotFoundError(f"Production checkpoint missing: {CURRENT_MODEL_PATH}")
    splits = load_splits(); train_images, train_masks = splits["train"]; val_images, val_masks = splits["validation"]
    report = validate_splits(splits, RESULTS_DIR / "dataset_quality_report.json")
    print("Dataset quality report:", {name: data.get("pairs") for name, data in report["splits"].items()})
    inspect_masks(train_masks); print(f"Fine-tuning {len(train_images)} train / {len(val_images)} validation samples")
    train_data, validation_data = make_dataset(train_images, train_masks, True), make_dataset(val_images, val_masks)
    model = tf.keras.models.load_model(CURRENT_MODEL_PATH, compile=False); history = {}
    freeze_encoder(model); compile_model(model, LEARNING_RATE)
    if FREEZE_ENCODER_EPOCHS:
        _extend(history, model.fit(train_data, validation_data=validation_data, epochs=FREEZE_ENCODER_EPOCHS, callbacks=_callbacks(), verbose=1))
    unfreeze_all(model); compile_model(model, FINE_TUNE_LEARNING_RATE)
    _extend(history, model.fit(train_data, validation_data=validation_data, initial_epoch=FREEZE_ENCODER_EPOCHS, epochs=EPOCHS, callbacks=_callbacks(), verbose=1))
    plot_history(history, RESULTS_DIR); _threshold_selection(tf.keras.models.load_model(IMPROVED_MODEL_PATH, compile=False), validation_data)
    print(f"Candidate model saved to {IMPROVED_MODEL_PATH}; production model remains {CURRENT_MODEL_PATH}")

if __name__ == "__main__": main()
