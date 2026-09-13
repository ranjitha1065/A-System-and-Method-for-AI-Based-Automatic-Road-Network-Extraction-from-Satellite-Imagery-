"""Lazy image decoding and paired TensorFlow augmentation."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import rasterio
import tensorflow as tf
from config import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED

def _path(value) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

def _read(path: str, mask: bool) -> np.ndarray:
    """Use Rasterio for TIFF/GeoTIFF and OpenCV for standard images/fallback."""
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            with rasterio.open(path) as source: return source.read(1) if mask else source.read()
        except (rasterio.errors.RasterioError, OSError, ValueError): pass
    flag = cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_UNCHANGED
    array = cv2.imread(path, flag)
    if array is None: raise ValueError(f"Could not read image: {path}")
    return array if mask else (np.moveaxis(array, -1, 0) if array.ndim == 3 else array[None, ...])

def _image_array(value) -> np.ndarray:
    data = _read(_path(value), False)
    if data.ndim != 3 or not data.shape[0]: raise ValueError("Image has no usable channels")
    if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2: data = np.concatenate((data, data[:1]), axis=0)
    dtype, image = data.dtype, np.moveaxis(data[:3], 0, -1).astype(np.float32)
    image = np.where(np.isfinite(image), image, 0)
    if dtype == np.uint8: image /= 255.0
    elif dtype == np.uint16: image /= 65535.0
    else: image = np.clip(image, 0, 1)
    return tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear").numpy().astype(np.float32)

def _mask_array(value) -> np.ndarray:
    raw = _read(_path(value), True)
    if raw.ndim == 3: raw = raw[0]
    mask = tf.image.resize(np.where(np.isfinite(raw), raw, 0).astype(np.float32)[..., None], (IMAGE_SIZE, IMAGE_SIZE), method="nearest").numpy()
    # Supports DeepGlobe 0/255 masks and already-binary TIFF labels.
    return (mask > (0.5 if float(np.max(raw)) <= 1.0 else 127.0)).astype(np.float32)

def inspect_masks(paths: list[str], samples: int = 3) -> None:
    for path in paths[:samples]:
        values = np.unique(_mask_array(path)).tolist()
        print(f"Mask {Path(path).name}: processed unique values={values}")
        if values not in ([0.0], [1.0], [0.0, 1.0]): raise ValueError(f"Mask is not binary: {path}")

def raw_pair_info(image_path: str | Path, mask_path: str | Path) -> dict[str, object]:
    """Read a pair without changing it; used by the dataset-quality audit."""
    image = _read(str(image_path), False)
    mask = _read(str(mask_path), True)
    image_shape = tuple(image.shape[-2:])
    mask_shape = tuple(mask.shape[-2:])
    finite_mask = np.where(np.isfinite(mask), mask, 0)
    threshold = 0.5 if float(np.max(finite_mask)) <= 1.0 else 127.0
    binary = finite_mask > threshold
    raw_values = np.unique(finite_mask)
    # Source labels may be 0/1 or 0/255. Any other value is reported, not modified.
    accepted_values = {0.0, 1.0, 255.0}
    return {
        "image_shape": image_shape, "mask_shape": mask_shape,
        "dimensions_match": image_shape == mask_shape,
        "mask_raw_values": raw_values[:16].tolist(),
        "mask_binary": bool(set(raw_values.astype(float).tolist()).issubset(accepted_values)),
        "road_coverage": float(binary.mean()),
    }

def _decode(image_path, mask_path):
    image, mask = tf.py_function(lambda a, b: (_image_array(a), _mask_array(b)), [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3)); mask.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return image, mask

def _augment(image, mask):
    """Apply shared geometry and image-only appearance transforms lazily."""
    if tf.random.uniform((), seed=RANDOM_SEED) > .5: image, mask = tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
    if tf.random.uniform((), seed=RANDOM_SEED) > .5: image, mask = tf.image.flip_up_down(image), tf.image.flip_up_down(mask)
    turns = tf.random.uniform((), 0, 4, tf.int32, seed=RANDOM_SEED); image, mask = tf.image.rot90(image, turns), tf.image.rot90(mask, turns)
    shift_y = tf.random.uniform((), -16, 17, tf.int32, seed=RANDOM_SEED); shift_x = tf.random.uniform((), -16, 17, tf.int32, seed=RANDOM_SEED)
    image, mask = tf.roll(image, (shift_y, shift_x), (0, 1)), tf.roll(mask, (shift_y, shift_x), (0, 1))
    scale = tf.random.uniform((), .9, 1.1, seed=RANDOM_SEED); scaled = tf.cast(IMAGE_SIZE * scale, tf.int32)
    image = tf.image.resize(tf.image.resize(image, (scaled, scaled)), (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    mask = tf.image.resize(tf.image.resize(mask, (scaled, scaled), method="nearest"), (IMAGE_SIZE, IMAGE_SIZE), method="nearest")
    image = tf.image.random_brightness(image, .10, seed=RANDOM_SEED); image = tf.image.random_contrast(image, .85, 1.15, seed=RANDOM_SEED)
    image = image + tf.random.normal(tf.shape(image), stddev=.015, seed=RANDOM_SEED)
    return tf.clip_by_value(image, 0., 1.), tf.cast(mask > .5, tf.float32)

def make_dataset(image_paths: list[str], mask_paths: list[str], training: bool = False, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    if not image_paths or len(image_paths) != len(mask_paths): raise ValueError("Image and mask paths must be equal, non-empty lists.")
    data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training: data = data.shuffle(len(image_paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    data = data.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training: data = data.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else: data = data.cache()
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_image_for_prediction(path: str | Path) -> np.ndarray: return _image_array(str(path))
def load_mask_for_prediction(path: str | Path) -> np.ndarray: return _mask_array(str(path))
