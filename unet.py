"""Existing U-Net architecture helpers for transfer learning."""
import tensorflow as tf
from config import IMAGE_SIZE, LEARNING_RATE
from utils import compile_model

def _block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs); x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x); x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

def build_unet() -> tf.keras.Model:
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="satellite_image"); skips = []; x = inputs
    for filters in (32, 64, 128, 256): x = _block(x, filters); skips.append(x); x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(.3)(_block(x, 512))
    for filters, skip in zip((256, 128, 64, 32), reversed(skips)): x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x); x = _block(tf.keras.layers.Concatenate()([x, skip]), filters)
    return compile_model(tf.keras.Model(inputs, tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="road_mask")(x), name="road_unet"), LEARNING_RATE)

def freeze_encoder(model: tf.keras.Model) -> None:
    """Freeze encoder/bottleneck layers; decoder starts after the first transpose convolution."""
    decoder_index = next((index for index, layer in enumerate(model.layers) if isinstance(layer, tf.keras.layers.Conv2DTranspose)), len(model.layers))
    for layer in model.layers[:decoder_index]: layer.trainable = False

def unfreeze_all(model: tf.keras.Model) -> None:
    for layer in model.layers: layer.trainable = True
