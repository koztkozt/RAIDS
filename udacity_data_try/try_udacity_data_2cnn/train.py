from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from config import TrainConfig


def create_comma_model_large_dropout(
    row, col, ch, path, load_weights=False
):  # change## parameter values // deepxplore Dave_dropout
    model = Sequential()

    # model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Conv2D(24, (3, 3), strides=(2, 2), padding="same", input_shape=(row, col, ch)))
    model.add(Activation("relu"))

    # model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Flatten())

    model.add(Dense(500))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Activation("relu"))
    model.add(Dense(20))
    model.add(Activation("relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    if load_weights:
        model.load_weights(path)

    print("Model is created and compiled..")
    return model
