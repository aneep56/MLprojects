import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import tkinter as tk
from tkinter import ttk

# Initialize the MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the MediaPipe Drawing module for visualization
mp_drawing = mp.solutions.drawing_utils

# Get the default audio endpoint (speaker)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))

# Initialize variables to keep track of previous thumb and index finger positions
prev_thumb_x, prev_thumb_y = 0, 0
prev_index_x, prev_index_y = 0, 0

# Create a function to update the volume
def update_volume(distance):
    min_distance, max_distance = 0.05, 0.3
    min_volume, max_volume = 0.0, 1.0  # Volume values should be between 0 and 1
    new_volume = (distance - min_distance) / (max_distance - min_distance) * (max_volume - min_volume) + min_volume

    # Set the system volume, ensuring it is within the valid range [min_volume, max_volume]
    new_volume = max(min_volume, min(max_volume, new_volume))
    volume.SetMasterVolumeLevelScalar(new_volume, None)

    return new_volume * 100  # Convert volume to percentage

# Create a GUI window for the volume control
root = tk.Tk()
root.title("Volume Control")
root.geometry("300x100")

# Create a Label to display volume control instructions
label = tk.Label(root, text="Adjust volume using hand gestures (index finger and thumb)")
label.pack()

# Create a Progressbar for volume control
volume_progress = ttk.Progressbar(root, orient='horizontal', length=200, mode='determinate')
volume_progress.pack()

# Create a function to update the volume Progressbar
def update_volume_progress():
    while True:
        distance = math.dist([prev_thumb_x, prev_thumb_y], [prev_index_x, prev_index_y])
        new_volume = update_volume(distance)
        new_volume = max(0, min(100, new_volume))  # Ensure the volume value is within the range [0, 100]
        volume_progress['value'] = new_volume
        root.update()

# Create a thread for updating volume Progressbar
import threading
volume_thread = threading.Thread(target=update_volume_progress)
volume_thread.daemon = True
volume_thread.start()

# Initialize the webcam capture
cap = cv2.VideoCapture(0)  # You can change the argument to use a different camera, e.g., 1 for an external webcam.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract coordinates of thumb and index finger
            thumb_x, thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_x, index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            prev_thumb_x, prev_thumb_y, prev_index_x, prev_index_y = thumb_x, thumb_y, index_x, index_y

            # Update the volume based on the distance
            distance = math.dist([thumb_x, thumb_y], [index_x, index_y])
            new_volume = update_volume(distance)

            # Draw the volume level on the camera feed
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 255, 255)  # Yellow color
            cv2.putText(frame, f'Volume: {new_volume:.2f}%', (10, 30), font, font_scale, font_color, 1)

    # Convert the frame for displaying in OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the camera feed
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Start the GUI main loop for volume control
root.mainloop()