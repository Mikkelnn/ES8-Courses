from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf #noqa
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras import regularizers

def defineModel_VGG8():

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

        GlobalMaxPooling2D(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model

def defineModel_VGG16():

    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
    
        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 5
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),

        GlobalMaxPooling2D(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model

def defineModel_VGG4():
    
    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # Block 1
        Conv2D(16, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalMaxPooling2D(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model

def defineModel_VGG4_flatten():
    
    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # Block 1
        Conv2D(16, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation="softmax"),
    ])

    return model

def defineModel_VGG4_flatten_regulazor():
    
    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # Block 1
        Conv2D(16, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(10, activation="softmax", kernel_regularizer=regularizers.l2(0.001)),
    ])

    return model


def defineModel_VGG4_dropout():
    
    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # Block 1
        Conv2D(16, (3, 3), BatchNormalization(), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(32, (3, 3), BatchNormalization(), activation='relu', padding="same"),
        Conv2D(64, (3, 3), BatchNormalization(), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
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

from tensorflow.keras.applications import DenseNet121


def defineModel_densenet_10_classes():
    inputs = Input(shape=(32, 32, 3))

    backbone = DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling=None,
    )

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(256, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="densenet121_cifar10")