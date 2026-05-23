%%writefile layer.py
# simplified safe version (prevents crashes)

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D

def DenseDDSSPPLayer(x, filters, rate):
    x = Conv2D(filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def DenseDDSSPP(x):
    for i in range(3):
        x = DenseDDSSPPLayer(x, 64, i)
    return x
