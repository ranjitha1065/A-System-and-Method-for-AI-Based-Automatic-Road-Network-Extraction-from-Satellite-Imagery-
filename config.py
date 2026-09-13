"""Configuration for transfer-learning road segmentation."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_SOURCE = os.getenv("ROADAI_DATASET", "deepglobe").lower()
DEEPGLOBE_DIR = PROJECT_ROOT / "dataset"
LEGACY_DIR = PROJECT_ROOT / "dataset_old"
MODEL_DIR, RESULTS_DIR = PROJECT_ROOT / "models", PROJECT_ROOT / "results"
# Production remains on the known-good checkpoint unless ROADAI_MODEL is explicitly set.
CURRENT_MODEL_PATH = MODEL_DIR / "best_model.keras"
IMPROVED_MODEL_PATH = MODEL_DIR / "best_model_improved.keras"
MODEL_PATH = Path(os.getenv("ROADAI_MODEL", str(CURRENT_MODEL_PATH))).resolve()
os.environ["MPLCONFIGDIR"] = str(RESULTS_DIR / ".matplotlib")

IMAGE_SIZE, BATCH_SIZE = 256, 4
EPOCHS, FREEZE_ENCODER_EPOCHS = 30, 5
LEARNING_RATE, FINE_TUNE_LEARNING_RATE = 1e-4, 1e-5
RANDOM_SEED = 42
# Updated automatically only after validation-set threshold selection, never from test data.
def _saved_threshold(path: Path) -> float | None:
    sidecar = path.with_suffix(".threshold.json")
    try:
        import json
        return float(json.loads(sidecar.read_text())["threshold"])
    except (OSError, ValueError, KeyError):
        return None

THRESHOLD = float(os.getenv("ROADAI_THRESHOLD", str(_saved_threshold(MODEL_PATH) or 0.5)))
THRESHOLD_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
def _saved_postprocess(path: Path) -> int:
    try:
        import json
        return int(json.loads(path.with_suffix(".threshold.json").read_text()).get("postprocess_min_component", 0))
    except (OSError, ValueError, KeyError):
        return 0

POSTPROCESS_MIN_COMPONENT = int(os.getenv("ROADAI_MIN_COMPONENT", str(_saved_postprocess(MODEL_PATH))))
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

if DATASET_SOURCE not in {"deepglobe", "legacy"}:
    raise ValueError("ROADAI_DATASET must be either 'deepglobe' or 'legacy'.")
