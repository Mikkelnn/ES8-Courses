from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf #noqa

def defineModel_image_10_classes():

    model = Sequential([
        Input(shape=(1024, 256, 1)),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 1)),  # (1024, 256) -> (512, 256)

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (512, 256) -> (256, 128)

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (256, 128) -> (128, 64)

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (128, 64) -> (64, 32)
        
        GlobalMaxPooling1D(),
        Dense(256, activation='relu'),
        Dense(10, activation="sigmoid"),
    ])

    return model