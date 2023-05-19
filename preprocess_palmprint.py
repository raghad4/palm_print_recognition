import cv2
import numpy as np


def preprocess_palmprint(image_path, ROI_width, ROI_height):
    """
    Preprocesses the palmprint image by extracting the ROI area of palmprint and resizing it to a fixed size.

    Args:
    - image_path: string representing the path to the palmprint image file
    - ROI_width: integer representing the width of the ROI area to extract
    - ROI_height: integer representing the height of the ROI area to extract

    Returns:
    - ROI: a numpy array representing the ROI area of palmprint
    """
    # Read the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median filtering to remove noise
    filtered = cv2.medianBlur(gray, 5)

    # Detect the hand contour and extract the bounding rectangle
    contours, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Extract the ROI area and resize it to a fixed size
    ROI = gray[y:y + h, x:x + w]
    ROI = cv2.resize(ROI, (ROI_width, ROI_height))

    # Normalize the pixel values to [0, 1]
    ROI = ROI.astype(np.float32) / 255.

    # Reshape the array to match the input shape of the CNN model
    ROI = ROI.reshape((1, ROI_height, ROI_width, 1))

    return ROI
