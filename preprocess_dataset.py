import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_dataset(input_dir, output_dir, ROI_width, ROI_height):
    """
    Preprocesses the palmprint dataset by extracting the ROI area from each image
    and saving the processed images to a new directory.

    Args:
        input_dir (str): Directory containing the original palmprint images.
        output_dir (str): Directory to save the processed palmprint images.
        ROI_width (int): Width of the ROI area to extract from each image.
        ROI_height (int): Height of the ROI area to extract from each image.
    """

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each identity directory in the input directory
    for identity_dir in os.listdir(input_dir):
        identity_path = os.path.join(input_dir, identity_dir)

        # Loop over each palmprint image in the identity directory
        for image_path in glob.glob(os.path.join(identity_path, '*.JPG')):
            # Load the palmprint image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Extract the ROI area from the palmprint image
            ROI_x = int(image.shape[1]/2 - ROI_width/2)
            ROI_y = int(image.shape[0]/2 - ROI_height/2)
            ROI = image[ROI_y:ROI_y+ROI_height, ROI_x:ROI_x+ROI_width]

            # Resize the ROI area to a fixed size
            ROI = cv2.resize(ROI, (ROI_width, ROI_height))

            # Save the processed image to the output directory
            identity_name = identity_dir.split('.')[0] # Get the identity name from the directory name
            image_name = os.path.basename(image_path) # Get the image name from the file path
            # put the image in same structure as the original dataset
            output_path = os.path.join(output_dir, identity_dir, image_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, ROI)


def load_dataset(input_dir, test_size=0.2, ROI_width=100, ROI_height=100):
    """
    Loads the palmprint dataset, preprocesses the images, and splits the dataset
    into training and testing sets.

    Args:
        input_dir (str): Directory containing the original palmprint images.
        test_size (float): Fraction of the dataset to use for testing (default 0.2).
        ROI_width (int): Width of the ROI area to extract from each image.
        ROI_height (int): Height of the ROI area to extract from each image.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """

    # Preprocess the dataset
    print("Preprocessing dataset...")
    output_dir = 'preprocessed_dataset'
    preprocess_dataset(input_dir, output_dir, ROI_width, ROI_height)

    # Load the preprocessed images
    print("Loading dataset...")
    X = []
    y = []
    for identity_dir in os.listdir(output_dir):
        identity_name = identity_dir.split('_')[0]
        identity_label = int(identity_name)
        for image_path in glob.glob(os.path.join(output_dir, identity_dir, '*.JPG')):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            X.append(image)
            y.append(identity_label)

    # Check if there are enough samples to create a training and testing set
    if len(X) == 0:
        raise ValueError("Dataset contains no images.")

    if len(X) == 1:
        X_train = np.array(X)
        X_test = np.array(X)
        y_train = np.array(y)
        y_test = np.array(y)

    else:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalize the images.
    X_train = (np.array(X_train) / 255) - 0.5
    X_test = (np.array(X_test) / 255) - 0.5
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Reshape the images.
    # Add channel dimension to images
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test
