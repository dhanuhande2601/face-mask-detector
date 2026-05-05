from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

face_model_path = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
prototxt_path = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
mask_model_path = os.path.join(BASE_DIR, "models", "mobilenet_model")

face_net = cv2.dnn.readNet(prototxt_path, face_model_path)
mask_model = load_model(mask_model_path)
# =========================
# HOME ROUTE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Decode base64 image
    image_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    (h, w) = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces_output = []  # ✅ important change

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            # ================= PREPROCESS =================
            face_resized = cv2.resize(face, (224, 224))
            face_resized = face_resized / 255.0
            face_resized = np.reshape(face_resized, (1, 224, 224, 3))

            # ================= PREDICT =================
            prediction = mask_model.predict(face_resized)[0]

            label = "Mask 😷" if np.argmax(prediction) == 0 else "No Mask ❌"
            confidence_score = round(np.max(prediction) * 100, 2)

            # ✅ SEND FULL DATA (IMPORTANT)
            faces_output.append({
                "box": [int(startX), int(startY), int(endX), int(endY)],
                "label": label,
                "confidence": confidence_score
            })

    return jsonify({"faces": faces_output})
# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)