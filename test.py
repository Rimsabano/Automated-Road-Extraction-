%%writefile test.py
import tensorflow as tf

from data_preprocessing import (
    load_dataset,
    split_dataset
)

from model import DeepLabV3PlusDenseDDSSPP

from utils import (
    CustomIoU,
    F1Score
)


# =========================
# PATHS
# =========================

IMAGE_DIR = "/content/data/images"
MASK_DIR = "/content/data/masks"

MODEL_PATH = "outputs/model_checkpoint.keras"

IMAGE_SIZE = 512


# =========================
# LOAD DATASET
# =========================

print("Loading Dataset...")

image_dataset, mask_dataset = load_dataset(
    IMAGE_DIR,
    MASK_DIR,
    size=IMAGE_SIZE,
    max_images=None
)


# =========================
# SPLIT DATASET
# =========================

_, _, X_test, _, _, y_test = split_dataset(
    image_dataset,
    mask_dataset
)


# =========================
# LOAD MODEL
# =========================

print("Loading Model...")

model = DeepLabV3PlusDenseDDSSPP(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

model.load_weights(MODEL_PATH)


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        CustomIoU(),
        F1Score()
    ]
)


# =========================
# EVALUATE
# =========================

print("Evaluating Model...")

results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Results")
print("Loss:", results[0])
print("Accuracy:", results[1])
print("IoU:", results[2])
print("F1 Score:", results[3])
