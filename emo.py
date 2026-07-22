import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

def mosaic_art(face_roi, block_size=15):
    """
    Create an artistic mosaic of the face region.
    Each block gets a random color sampled from the original area.
    """
    (h, w, _) = face_roi.shape
    mosaic = np.zeros_like(face_roi)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            # Randomly sample a pixel color from within the block
            rand_y = random.randint(y, y_end - 1)
            rand_x = random.randint(x, x_end - 1)
            color = face_roi[rand_y, rand_x]

            mosaic[y:y_end, x:x_end] = color

    # Add artistic tint
    overlay = np.full_like(mosaic, (random.randint(0,255),
                                    random.randint(0,255),
                                    random.randint(0,255)))
    blended = cv2.addWeighted(mosaic, 0.8, overlay, 0.2, 0)
    return blended

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size > 0:
                # Apply artistic mosaic
                art_face = mosaic_art(face_roi, block_size=15)
                frame[y1:y2, x1:x2] = art_face

    cv2.imshow("🎨 Face Mosaic Art", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
