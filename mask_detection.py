import urllib.request
import os

if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector model (Caffe)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Load mask detector model (.h5 format required)
mask_net = load_model("mask_detector.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure coordinates are within the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            # Preprocess face for prediction
            face_input = cv2.resize(face, (224, 224))
            face_input = face_input.astype("float") / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            # Predict mask / no mask
            (mask, no_mask) = mask_net.predict(face_input, verbose=0)[0]
            label = "Mask" if mask > no_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display label + bounding box
            cv2.putText(frame, f"{label}: {max(mask, no_mask)*100:.2f}%", 
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show result
    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
