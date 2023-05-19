import numpy as np
from tensorflow.python.debug.examples.debug_mnist import tf
from tensorflow.python.keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from detect_palmprint import detect_palmprint

if __name__ == '__main__':
    # Load the preprocessed data from files
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)

    # test model with X_test, y_test
    model = tf.keras.models.load_model('model.h5')

    label_name = detect_palmprint(model_path='model.h5',
                        image_path='test/001_F_R_0.JPG',
                        ROI_width=400,
                        ROI_height=400,
                     label_names=to_categorical(y_test))

    print("Detected palmprint identity:", label_name)

    # Predict on the first 5 test images.
    predictions = model.predict(X_test[:5])

    # Print our model's predictions.
    print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

    # Print the actual labels
    print(y_test[:5])  # [7, 2, 1, 0, 4]

    # Print accuracy score as percentage
    print( model.evaluate(X_test, to_categorical(y_test)))

    # Check our predictions against the ground truths.
    print(to_categorical(y_test)[:5])  # [7, 2, 1, 0, 4]

