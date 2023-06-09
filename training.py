import cv2  # Library for computer vision tasks, such as face recognition
import numpy as np  # Library for numerical calculations and working with matrices
from PIL import Image  # Library for image processing
import os  # Library for file and directory operations

# Specify the path to the dataset
path = 'dataset'

# Create an instance of the LBPH (Local Binary Patterns Histograms) face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar cascade classifier for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    """
    Function to extract images and their corresponding labels from a given directory.

    Parameters:
    - path: Path to the directory containing the images

    Returns:
    - facesamples: List of face images
    - ids: List of corresponding labels for each face image
    """
    # Get the list of image paths in the specified directory
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize lists to store face samples and their corresponding labels
    facesamples = []
    ids = []

    # Iterate over each image path
    for imagePath in imagePath:
        # Open the image using PIL and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
#         In the line PIL_img = Image.open(imagePath).convert('L'), the 'L' parameter is used in the convert() method to convert the image to grayscale mode.
# In the context of PIL (Python Imaging Library), the mode 'L' refers to grayscale or black-and-white images. Grayscale images have a single channel that represents the intensity or brightness of each pixel. The pixel values range from 0 (black) to 255 (white), with intermediate values representing different shades of gray.
# By converting the image to grayscale mode, each pixel's color information is discarded, and only the intensity values are retained. This simplifies image processing tasks that do not require color information, such as face recognition, as the focus is primarily on the shape and texture of the face rather than its color.
# Converting the image to grayscale using 'L' mode is a common preprocessing step in many computer vision applications to reduce the complexity of the data and improve the efficiency of subsequent algorithms or operations that work with grayscale images.

        # Convert the PIL image to a NumPy array
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract the label from the image file name
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Detect faces in the image using the Haar cascade classifier
        faces = detector.detectMultiScale(img_numpy)

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (face) from the image
            facesamples.append(img_numpy[y:y+h, x:x+w])

            # Add the corresponding label to the labels list
            ids.append(id)

    # Return the extracted face samples and their labels
    return facesamples, ids

# Print a message indicating that face training is in progress
print("\n[INFO] Training Faces...")

# Extract face samples and their corresponding labels from the dataset
faces, ids = getImagesAndLabels(path)

# Train the face recognizer using the extracted face samples and labels
recognizer.train(faces, np.array(ids))

# Save the trained model to a file
recognizer.write('trainer/trainer.yml')

# Print a message indicating the number of faces trained
print("\n[INFO] {0} faces trained.".format(len(np.unique(ids))))
