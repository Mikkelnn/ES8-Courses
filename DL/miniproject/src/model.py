from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf #noqa
from tensorflow.keras.applications import ConvNeXtTiny

def defineModel_image_10_classes():

    model = Sequential([
        Input(shape=(32, 32, 3)),
        
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
        Dense(256, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model

def defineModel_convnext_10_classes():
    inputs = Input(shape=(32, 32, 3))

    backbone = ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling=None,
    )

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(256, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="convnext_tiny_cifar10")