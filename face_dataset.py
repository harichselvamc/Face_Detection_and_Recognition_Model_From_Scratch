import cv2  # Library for computer vision tasks, such as face recognition
import os  # Library for file and directory operations

# Create a VideoCapture object to capture video from the default camera (index 0)
cam = cv2.VideoCapture(0)

# Set the width and height of the captured video frames
cam.set(3, 640)  # Set video width
# This line sets the video width of the capture object cam to 640 pixels. The first parameter 3 is the property ID for video width, and the second parameter 640 is the desired width value in pixels. Adjusting the video width determines the horizontal size of the captured video frames.
cam.set(4, 480)  # Set video height
# This line sets the video height of the capture object cam to 480 pixels. The first parameter 4 is the property ID for video height, and the second parameter 480 is the desired height value in pixels. Adjusting the video height determines the vertical size of the captured video frames.

# Load the Haar cascade classifier for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prompt the user to enter a numeric face id for the person being captured
face_id = input('\n Enter user id: ')

print("\n [INFO] Initializing face capture....")
# Initialize individual sampling face count
count = 0

# Start an infinite loop to continuously capture video frames
while True:
    # Read the current frame from the video capture
    ret, img = cam.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image using the Haar cascade classifier
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face on the original colored frame
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder with a unique filename
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        # Display the original frame with detected faces
        cv2.imshow('image', img)

    # Wait for 100 milliseconds and check if the 'Esc' key is pressed
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # Break the loop if the desired number of face samples is reached (e.g., 30)
    elif count >= 90:
        break

# Clean up resources
print("\n [INFO] Exiting Program")
cam.release()  # Release the camera
cv2.destroyAllWindows()  # Close all open windows
