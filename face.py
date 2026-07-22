import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

def get_emotion(landmarks, img_w, img_h):
    # Get key facial landmark indices for mouth
    top_lip = landmarks[13]   # upper lip
    bottom_lip = landmarks[14]  # lower lip
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Convert normalized coordinates to pixels
    top_lip_y = int(top_lip.y * img_h)
    bottom_lip_y = int(bottom_lip.y * img_h)
    left_eye_x = int(left_eye.x * img_w)
    right_eye_x = int(right_eye.x * img_w)

    # Calculate mouth openness & eye width
    mouth_open = abs(bottom_lip_y - top_lip_y)
    eye_width = abs(right_eye_x - left_eye_x)

    # Define thresholds for simple emotion detection
    if mouth_open > 25:
        return "Surprised"
    elif mouth_open > 10:
        return "Smiling"
    else:
        return "Neutral"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            emotion = get_emotion(face_landmarks.landmark, img_w, img_h)

            # Color feedback based on emotion
            if emotion == "Smiling":
                color = (0, 255, 0)
            elif emotion == "Surprised":
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            # Draw face outline
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(frame, (x, y), 1, color, -1)

            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Emotion-Aware Face Detection", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
