from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def create_model(height, width, num_classes):
    """
    Creates a CNN model using Densenet architecture for palmprint recognition.

    Args:
    - height: integer representing the height of the input image
    - width: integer representing the width of the input image
    - num_classes: integer representing the number of classes (palmprint identities) to be recognized

    Returns:
    - model: a compiled CNN model for palmprint recognition
    """
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes+1, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
