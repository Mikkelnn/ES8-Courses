import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

class Data():

    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.cifar10.load_data()

        self.X_train = self.X_train.astype('float32') / 255.0 # scale to [0, 1]
        self.X_test = self.X_test.astype('float32') / 255.0
        self.y_train = to_categorical(self.y_train, 10) #Convert integer class vector to binary class matrix
        self.y_test = to_categorical(self.y_test, 10)

        self.CLASS_LABELS = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

    def visualize(self) -> None:
        plt.figure(figsize=(10,10))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.X_train[i])
            plt.xlabel(self.CLASS_LABELS[np.argmax(self.y_train[i])])
        plt.show()