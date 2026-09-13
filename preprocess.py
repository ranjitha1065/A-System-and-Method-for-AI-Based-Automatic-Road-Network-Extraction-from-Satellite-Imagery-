"""Dataset-layout aware pairing without loading imagery into memory."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from config import DATASET_SOURCE, DEEPGLOBE_DIR, LEGACY_DIR, RANDOM_SEED, SUPPORTED_EXTENSIONS
from dataset import raw_pair_info


def _key(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_sat", "_image", "_img", "_mask", "_label", "_labels"):
        if stem.endswith(suffix): return stem[: -len(suffix)]
    return stem


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir(): raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files: raise ValueError(f"No supported image files found in: {directory}")
    return files


def image_files(directory: str | Path) -> list[str]:
    """Return satellite/image files from an unlabeled inference folder."""
    return [str(path) for path in _files(Path(directory)) if not any(token in path.stem.lower() for token in ("_mask", "_label"))]


def pair_files(image_directory: str | Path, mask_directory: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Strictly pair images and masks by normalized filename stem.

    DeepGlobe stores `id_sat.jpg` and `id_mask.png` in the same folder;
    legacy data stores images and labels in separate directories. Both work.
    """
    image_directory = Path(image_directory); mask_directory = Path(mask_directory) if mask_directory else image_directory
    image_files = _files(image_directory); mask_files = _files(mask_directory)
    images = {_key(path): path for path in image_files if not any(token in path.stem.lower() for token in ("_mask", "_label"))}
    masks = {_key(path): path for path in mask_files if any(token in path.stem.lower() for token in ("_mask", "_label"))}
    if mask_directory != image_directory:
        masks = {_key(path): path for path in mask_files}
    missing_masks, missing_images = sorted(images.keys() - masks.keys()), sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        parts = []
        if missing_masks: parts.append(f"{len(missing_masks)} images have no mask (e.g. {missing_masks[0]})")
        if missing_images: parts.append(f"{len(missing_images)} masks have no image (e.g. {missing_images[0]})")
        raise ValueError("Image-mask alignment failed: " + "; ".join(parts))
    keys = sorted(images)
    if not keys: raise ValueError("No valid image-mask pairs found.")
    return [str(images[key]) for key in keys], [str(masks[key]) for key in keys]


def load_splits(source: str = DATASET_SOURCE) -> dict[str, tuple[list[str], list[str]]]:
    """Return train/validation/test path pairs for DeepGlobe or legacy layouts."""
    if source == "deepglobe":
        train_images, train_masks = pair_files(DEEPGLOBE_DIR / "train")
        valid_folder, test_folder = DEEPGLOBE_DIR / "valid", DEEPGLOBE_DIR / "test"
        # Official DeepGlobe validation/test downloads may contain imagery only.
        # Create a reproducible labelled validation holdout from train in that case.
        try:
            validation_images, validation_masks = pair_files(valid_folder)
        except ValueError:
            indices = np.random.default_rng(RANDOM_SEED).permutation(len(train_images)); count = max(1, round(.15 * len(train_images)));
            validation_indices, train_indices = indices[:count], indices[count:]
            validation_images = [train_images[index] for index in validation_indices]; validation_masks = [train_masks[index] for index in validation_indices]
            train_images = [train_images[index] for index in train_indices]; train_masks = [train_masks[index] for index in train_indices]
            print("DeepGlobe valid labels are unavailable; using a deterministic 15% labelled train holdout.")
        splits = {"train": (train_images, train_masks), "validation": (validation_images, validation_masks), "test": (image_files(test_folder), [])}
    elif source == "legacy":
        layout = {"train": ("train", "train_labels"), "validation": ("validation", "validation_labels"), "test": ("test", "test_labels")}
        splits = {name: pair_files(LEGACY_DIR / "images" / image, LEGACY_DIR / "masks" / mask) for name, (image, mask) in layout.items()}
    else: raise ValueError(f"Unknown dataset source: {source}")
    for name, (images, masks) in splits.items(): print(f"{source.title()} {name}: {len(images)} {'paired' if masks else 'unlabeled'} samples")
    return splits


def load_dataset(image_dir: str | Path, mask_dir: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Compatibility wrapper retained for existing scripts."""
    return pair_files(image_dir, mask_dir)


def validate_splits(splits: dict[str, tuple[list[str], list[str]]], output_path: str | Path, max_samples: int | None = None) -> dict:
    """Create a read-only paired-data quality report and stop on unsafe training data."""
    report: dict[str, object] = {"splits": {}, "errors": [], "warnings": []}
    for split, (images, masks) in splits.items():
        if not masks:
            report["splits"][split] = {"pairs": len(images), "labelled": False}
            continue
        if len(images) != len(masks):
            report["errors"].append(f"{split}: image/mask count mismatch")
            continue
        entries, coverage = [], []
        pairs = zip(images, masks) if max_samples is None else zip(images[:max_samples], masks[:max_samples])
        for image, mask in pairs:
            try:
                info = raw_pair_info(image, mask)
                entries.append({"image": Path(image).name, "mask": Path(mask).name, **info})
                coverage.append(info["road_coverage"])
                if not info["dimensions_match"]: report["errors"].append(f"{split}: dimension mismatch: {Path(image).name}")
                if not info["mask_binary"]: report["errors"].append(f"{split}: non-binary processed mask: {Path(mask).name}")
            except Exception as error:
                report["errors"].append(f"{split}: unreadable pair {Path(image).name}: {error}")
        mean_coverage = float(np.mean(coverage)) if coverage else 0.0
        if mean_coverage > .50: report["warnings"].append(f"{split}: mean road coverage {mean_coverage:.1%}; masks may be inverted.")
        report["splits"][split] = {"pairs": len(images), "checked": len(entries), "labelled": True, "mean_road_coverage": mean_coverage, "samples": entries[:10]}
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(report, indent=2))
    if report["errors"]: raise ValueError("Dataset validation failed; see " + str(destination))
    return report
