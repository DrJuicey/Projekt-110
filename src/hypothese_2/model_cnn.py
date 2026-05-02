import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers



# Baseline Model
def cnn_base():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),                # MNIST images: 28x28 grayscale
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((3, 3)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")          # 10 classes (digits 0-9)
    ])
    return model


# CNN with Dropout regularization.
def cnn_dropout():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((3, 3)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),                            # 50% of neurons randomly dropped
        layers.Dense(10, activation="softmax")
    ])
    return model


# CNN with L2 (weight decay) regularization.
def cnn_l2():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D( 
            32, (3, 3), activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),                                              # L2 regularization applied to convolutional layer
        layers.MaxPooling2D((3, 3)),
        layers.Flatten(),
        layers.Dense(
            64, activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dense(10, activation="softmax") #
    ])
    return model


# CNN trained with data augmentation 
def cnn_aug():
    return cnn_base()


# Backward-compatible aliases for any existing imports.
def build_base_model():
    return cnn_base()


def build_dropout_model():
    return cnn_dropout()


def build_weight_decay_model():
    return cnn_l2()


def build_augmented_model():
    return cnn_aug()
