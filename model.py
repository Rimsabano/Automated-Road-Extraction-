%%writefile model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception

def DeepLabV3PlusDenseDDSSPP(input_shape=(512,512,3)):

    inputs = Input(shape=input_shape)

    base = Xception(weights="imagenet", include_top=False, input_tensor=inputs)

    x = base.output

    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D((4,4))(x)

    x = Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inputs, x)
