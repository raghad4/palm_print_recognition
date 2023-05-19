import numpy as np
from tensorflow.python.debug.examples.debug_mnist import tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from create_model import create_model
from preprocess_dataset import load_dataset
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ROI_width = 400
    ROI_height = 400
    EPOCHS = 10

    X_train, X_test, y_train, y_test = load_dataset(input_dir='dataset', test_size=0.2, ROI_width=ROI_width, ROI_height=ROI_height)

    # Plot the first image in the dataset
    plt.imshow(X_train[1], cmap='gray')
    # plt.imshow(X_train[0])
    plt.show()


    # Save the preprocessed data to files
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

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

    # train model with X_train, y_train
    model = create_model(
        height=ROI_height,
        width=ROI_width,
        num_classes=len(np.unique(y_train))
    )
    model.summary()

    try:
        print("Starting training...")

        model.fit(
            X_train,
            to_categorical(y_train),
            epochs=EPOCHS,
            batch_size=32,
            validation_data=(X_test, to_categorical(y_test))
        )

    except Exception as e:
        print(e)
        raise e

    print("Done training!")

    # save model
    model.save('model.h5')
    model.save_weights('model_weights.h5')

    # test model with X_test, y_test
    model = tf.keras.models.load_model('model.h5')
    # model.evaluate(X_test, y_test)
    model.evaluate(X_test, to_categorical(y_test))

    # Predict on the first 5 test images.
    predictions = model.predict(X_test[:5])

    # Print our model's predictions.
    print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

    # Check our predictions against the ground truths.
    print(to_categorical(y_test)[:5])  # [7, 2, 1, 0, 4]

    print("Done!")
