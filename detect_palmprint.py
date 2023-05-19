import numpy as np
from tensorflow.keras.models import load_model

from preprocess_palmprint import preprocess_palmprint


def detect_palmprint(model_path, image_path, ROI_width, ROI_height, label_names):
    """
    Detects the person from a palmprint image using a trained CNN model.

    Args:
    - model_path: string representing the path to the saved model file
    - image_path: string representing the path to the palmprint image file
    - ROI_width: integer representing the width of the ROI area to extract
    - ROI_height: integer representing the height of the ROI area to extract
    - label_names: list of strings representing the names of the palmprint identities

    Returns:
    - label_name: string representing the name of the detected palmprint identity
    """
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the palmprint image and extract the ROI area of palmprint
    ROI = preprocess_palmprint(image_path, ROI_width, ROI_height)

    # Reshape the ROI area of palmprint to match the input shape of the CNN model
    ROI = np.expand_dims(ROI, axis=-1)

    # Use the trained CNN model to predict the palmprint identity
    prediction = model.predict(ROI)
    predicted_label = np.argmax(prediction)

    # Get the name of the detected palmprint identity
    label_name = label_names[predicted_label]

    return label_name
