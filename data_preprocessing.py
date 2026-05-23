%%writefile data_preprocessing.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_dataset(image_dir, mask_dir, size=512, max_images=None):
    image_dataset, mask_dataset = [], []

    image_subdirs = sorted(os.listdir(image_dir))

    if max_images:
        image_subdirs = image_subdirs[:max_images]

    for subdir in image_subdirs:
        img_folder = os.path.join(image_dir, subdir)
        mask_folder = os.path.join(mask_dir, subdir)

        image_files = sorted(os.listdir(img_folder))
        mask_files = sorted(os.listdir(mask_folder))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(img_folder, img_file)
            mask_path = os.path.join(mask_folder, mask_file)

            image = Image.open(img_path).convert("RGB").resize((size, size))
            mask = Image.open(mask_path).convert("L").resize((size, size))

            image_dataset.append(np.array(image) / 255.0)
            mask_dataset.append(np.array(mask) / 255.0)

    return np.array(image_dataset), np.expand_dims(np.array(mask_dataset), axis=-1)


def split_dataset(X, Y, test_size=0.2, val_size=0.1):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size/(1-test_size),
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
