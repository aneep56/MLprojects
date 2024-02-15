import cv2
import numpy as np
from pyfirmata import Arduino, util
import time
import threading

# Connect to Arduino
board = Arduino('COM8')  # Change 'COM3' to the port your Arduino is connected to
it = util.Iterator(board)
it.start()

# Define pin for buzzer
buzzer_pin = board.get_pin('d:11:p')  # Change 'd:3:p' to the pin you have connected the buzzer to

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# Set video capture properties for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set a threshold for black color detection
black_threshold = 70  # Adjust this threshold as needed

# Variable to keep track of the person count
person_count = 0

# Function to detect faces in a separate thread
def detect_faces():
    global person_count

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Reset the person count for each frame
        person_count = 0

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            person_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Person {person_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Get the region of interest (ROI) for the detected face
            roi_gray = gray[y:y + h, x:x + w]

            # Calculate the percentage of black pixels in the ROI
            black_percentage = np.count_nonzero(roi_gray < black_threshold) / (w * h) * 100

            # If black clothing is detected, activate the buzzer
            if black_percentage > 60:  # Adjust this threshold as needed
                print("Black clothing detected! Activating buzzer...")
                buzzer_pin.write(1)
                time.sleep(0)  # Buzzer active for 1 second
                buzzer_pin.write(0)

        # Display the frame with face detection
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the face detection thread
face_detection_thread = threading.Thread(target=detect_faces)
face_detection_thread.start()

# Wait for the face detection thread to finish
face_detection_thread.join()

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
