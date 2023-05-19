import numpy as np
from tensorflow.keras.datasets import mnist as mnists
from matplotlib import pyplot as plt

import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


def load_dataset():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnists.load_data()

    # Plot the first image in the dataset
    plt.imshow(X_train[0], cmap='gray')
    plt.show()

    # Print the shape of the data
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Print the number of training and testing samples
    print("Number of training samples:", len(X_train))
    print("Number of testing samples:", len(X_test))

    # Print the shape of the labels
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Print the number of training and testing labels
    print("Number of training labels:", len(y_train))
    print("Number of testing labels:", len(y_test))

    # Print the number of classes
    print("Number of classes:", len(np.unique(y_train)))

    return X_train, X_test, y_train, y_test


def create_model():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # Plot the first image in the dataset
    plt.imshow(train_images[3], cmap='gray')
    plt.show()

    # Normalize the images.
    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Reshape the images.
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2

    # Build the model.
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=3,
        validation_data=(test_images, to_categorical(test_labels)),
    )


if __name__ == '__main__':
    # load_dataset()
    create_model()
