import cv2
import pickle
import numpy as np

# Load Trained Model
model_path = 'D:/VScode/CV_WORKSHOP/face_recognition_model.pkl'

with open(model_path, 'rb') as f:
    pipe, label_names = pickle.load(f)

# Webcam Setup
cap = cv2.VideoCapture(0)  # 0 for default camera, try 1 if using external webcam

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_size = (128, 128)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, face_size)
        face_flat = face_resized.flatten().reshape(1, -1)

        # Predict
        pred = pipe.predict(face_flat)
        label = label_names[pred[0]]

        # Label
        label_text = f"{label}"

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Draw bounding box
        cv2.rectangle(frame, (x, y, x + w, y + h), (0, 255, 0), 2)

        # Draw green background for label
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            (0, 255, 0),
            thickness=-1
        )

        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()