import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

face_model_path = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
prototxt_path = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")

mask_model_path = os.path.join(BASE_DIR, "models", "mobilenet_model.h5")

# =========================
# LOAD MODELS
# =========================

face_net = cv2.dnn.readNet(prototxt_path, face_model_path)
mask_model = load_model(mask_model_path)

# =========================
# LOAD IMAGE
# =========================

image_path = os.path.join(BASE_DIR, "test.jpg")  # put your image here
frame = cv2.imread(image_path)

(h, w) = frame.shape[:2]

# =========================
# FACE DETECTION
# =========================

blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))

face_net.setInput(blob)
detections = face_net.forward()

# =========================
# LOOP OVER FACES
# =========================

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")

        # Extract face
        face = frame[startY:endY, startX:endX]

        if face.size == 0:
            continue

        # =========================
        # PREPROCESS FACE
        # =========================

        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        # =========================
        # PREDICT
        # =========================

        prediction = mask_model.predict(face)[0]

        label = "Mask" if np.argmax(prediction) == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # =========================
        # DRAW RESULT
        # =========================

        text = f"{label}: {max(prediction)*100:.2f}%"

        cv2.putText(frame, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.rectangle(frame, (startX, startY),
                      (endX, endY), color, 2)

# =========================
# SHOW OUTPUT
# =========================

cv2.imshow("Mask Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()