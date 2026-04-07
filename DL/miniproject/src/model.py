from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow as tf #noqa

def defineModel_image_10_classes():

    model = Sequential([
        Input(shape=(32, 32, 1)),
        
        # Block 1
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
    
        # Block 3
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalMaxPooling2D(),
        Dense(256, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model