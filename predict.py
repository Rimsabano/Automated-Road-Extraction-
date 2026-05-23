%%writefile predict.py
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from model import DeepLabV3PlusDenseDDSSPP


# =========================
# SETTINGS
# =========================

IMAGE_SIZE = 512

MODEL_PATH = "outputs/model_checkpoint.keras"

INPUT_IMAGE_PATH = "sample.jpg"

OUTPUT_DIR = "outputs"

OUTPUT_MASK_PATH = os.path.join(
    OUTPUT_DIR,
    "predicted_mask.png"
)

OVERLAY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "overlay_output.png"
)


# =========================
# CREATE OUTPUT DIRECTORY
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = DeepLabV3PlusDenseDDSSPP(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

model.load_weights(MODEL_PATH)

print("Model loaded successfully")


# =========================
# LOAD IMAGE
# =========================

print("Loading image...")

original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")

original_size = original_image.size

image = original_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)

image_array = np.array(image) / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)


# =========================
# PREDICTION
# =========================

print("Predicting roads...")

prediction = model.predict(image_array)[0]


# =========================
# PROCESS MASK
# =========================

mask = (prediction > 0.5).astype(np.uint8)

mask = mask.squeeze() * 255

mask_image = Image.fromarray(mask)

mask_image = mask_image.resize(original_size)

mask_image.save(OUTPUT_MASK_PATH)

print("Mask saved successfully")


# =========================
# CREATE OVERLAY
# =========================

original_cv = cv2.cvtColor(
    np.array(original_image),
    cv2.COLOR_RGB2BGR
)

mask_cv = cv2.imread(
    OUTPUT_MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

colored_mask = np.zeros_like(original_cv)

colored_mask[:, :, 1] = mask_cv

overlay = cv2.addWeighted(
    original_cv,
    0.7,
    colored_mask,
    0.3,
    0
)

cv2.imwrite(
    OVERLAY_OUTPUT_PATH,
    overlay
)

print("Overlay image saved successfully")


# =========================
# DONE
# =========================

print("\nPrediction Completed Successfully")
print("Mask Path:", OUTPUT_MASK_PATH)
print("Overlay Path:", OVERLAY_OUTPUT_PATH)
