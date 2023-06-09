import cv2  # Importing the OpenCV library for computer vision tasks
import numpy as np  # Importing NumPy for numerical calculations
import os  # Importing the operating system module

# Create an instance of the LBPH (Local Binary Patterns Histograms) face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained recognizer model from the file
recognizer.read('trainer/trainer.yml')

# Path to the Haar cascade classifier for face detection
cascadePath = "haarcascade_frontalface_default.xml"

# Create an instance of the Haar cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cascadePath)

# Specify the font for displaying text on the image
font = cv2.FONT_HERSHEY_TRIPLEX

# Initialize the ID counter
id = 0

# List of names corresponding to each recognized face
names = [0, 1, 2, 3, 'Z', 'W']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define the minimum window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    # Read the current frame from the video capture
    ret, img = cam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar cascade classifier
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face on the original colored frame
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Use the recognizer to predict the ID and confidence of the recognized face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if the confidence is less than 100, which indicates a perfect match
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Put the recognized ID and confidence as text on the image
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    # Display the image with the recognized faces
    cv2.imshow('camera', img)

    # Wait for a key press and check if the 'Esc' key is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Clean up resources
print("\n [INFO] Exiting Program")
cam.release()  # Release the camera
cv2.destroyAllWindows()  # Close all open windows
