from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

class Data():

    def __init__(self, val_split = 0.20):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()

        val_length = int(val_split * len(self.x_train))
        
        self.x_train = self.x_train.astype('float32') / 255.0 # scale to [0, 1]
        self.y_train = to_categorical(self.y_train, 10) #Convert integer class vector to binary class matrix

        self.x_val = self.x_train[-val_length:] 
        self.y_val = self.y_train[-val_length:]

        self.x_train = self.x_train[:-val_length]
        self.y_train = self.y_train[:-val_length]

        print(f"Training set: {self.x_train.shape}, {self.y_train.shape}")
        print(f"Validation set: {self.x_val.shape}, {self.y_val.shape}")

        self.x_test = self.x_test.astype('float32') / 255.0
        self.y_test = to_categorical(self.y_test, 10)

        self.CLASS_LABELS = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

    def visualize(self) -> None:
        plt.figure(figsize=(10,10))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i])
            plt.xlabel(self.CLASS_LABELS[np.argmax(self.y_train[i])])
        plt.show()

if __name__ == '__main__':
    data = Data()
    # data.visualize()