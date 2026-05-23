%%writefile train.py
import os
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from data_preprocessing import (
    load_dataset,
    split_dataset
)

from model import DeepLabV3PlusDenseDDSSPP
from utils import CustomIoU, F1Score


# =========================
# PATHS
# =========================

IMAGE_DIR = "/contents/sample_data/content/data/images"
MASK_DIR = "/content/sample_data/content/data/masks"

MODEL_SAVE_PATH = "outputs/model_checkpoint.keras"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50


# =========================
# CREATE OUTPUT DIRECTORY
# =========================

os.makedirs("outputs", exist_ok=True)


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

image_dataset, mask_dataset = load_dataset(
    IMAGE_DIR,
    MASK_DIR,
    size=IMAGE_SIZE,
    max_images=None
)

print("Dataset Loaded Successfully")

print("Images Shape:", image_dataset.shape)
print("Masks Shape:", mask_dataset.shape)


# =========================
# SPLIT DATASET
# =========================

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    image_dataset,
    mask_dataset
)


# =========================
# BUILD MODEL
# =========================

print("Building Model...")

model = DeepLabV3PlusDenseDDSSPP(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)


# =========================
# OPTIMIZER
# =========================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        CustomIoU(),
        F1Score()
    ]
)

model.summary()


# =========================
# CALLBACKS
# =========================

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)


# =========================
# TRAIN MODEL
# =========================

print("Training Started...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

print("Training Completed")
