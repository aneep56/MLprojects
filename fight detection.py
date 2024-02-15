import cv2
import mediapipe as mp
from pyfirmata import Arduino, util
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize VideoCapture with low resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Previous landmark positions
prev_landmarks = None

# Flags to indicate if punching or kicking action was previously detected
punching_detected = False
kicking_detected = False

# Thresholds for detecting punching and kicking actions
PUNCH_THRESHOLD = 20
KICK_THRESHOLD = 20

# Cooldown periods (in frames) for punching and kicking actions
PUNCH_COOLDOWN = 30
KICK_COOLDOWN = 30

# Cooldown counters
punch_cooldown_counter = 0
kick_cooldown_counter = 0

# Connect to Arduino board
board = Arduino('COM8')

# Define buzzer pin
buzzer_pin = board.get_pin('d:11:o')

# Start iterator thread
it = util.Iterator(board)
it.start()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to trigger buzzer
def trigger_buzzer():
    buzzer_pin.write(1)
    time.sleep(0.5)
    buzzer_pin.write(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(rgb_frame)
    
    # Draw skeleton on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract current landmark positions
        landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in results.pose_landmarks.landmark]
        
        # Analyze human activities if landmarks for kicking and punching are available
        if prev_landmarks is not None and len(landmarks) > max(mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value):
            # Calculate changes in key landmark positions
            left_wrist_movement = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1] - prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]
            right_wrist_movement = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1] - prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]
            left_knee_movement = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0] - prev_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0]
            right_knee_movement = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][0] - prev_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][0]
            
            # Punching detection
            if not punching_detected and (left_wrist_movement > PUNCH_THRESHOLD or right_wrist_movement > PUNCH_THRESHOLD):
                punching_detected = True
                punch_cooldown_counter = PUNCH_COOLDOWN
                trigger_buzzer()
            elif punching_detected and punch_cooldown_counter > 0:
                punch_cooldown_counter -= 1
            elif punching_detected and punch_cooldown_counter == 0:
                punching_detected = False
            
            # Kicking detection
            if not kicking_detected and (left_knee_movement > KICK_THRESHOLD or right_knee_movement > KICK_THRESHOLD):
                kicking_detected = True
                kick_cooldown_counter = KICK_COOLDOWN
                trigger_buzzer()
            elif kicking_detected and kick_cooldown_counter > 0:
                kick_cooldown_counter -= 1
            elif kicking_detected and kick_cooldown_counter == 0:
                kicking_detected = False
        
        # Update previous landmarks
        prev_landmarks = landmarks
    
    # Perform face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw yellow rectangles around detected faces and count the number of persons
    num_persons = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        num_persons += 1
    
    # Display the number of persons
    cv2.putText(frame, f'Persons: {num_persons}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Human Activity Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
