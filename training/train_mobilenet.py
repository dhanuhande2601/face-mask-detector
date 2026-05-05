import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =========================
# 1. LOAD DATASET
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset")

categories = ["with_mask", "without_mask"]

data = []
labels = []

for category in categories:
    path = os.path.join(dataset_path, category)

    if not os.path.exists(path):
        print("❌ Folder not found:", path)
        continue

    label = categories.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (224, 224))

            data.append(image)
            labels.append(label)

        except Exception as e:
            print("Skipping:", img, e)

print("✅ Total images loaded:", len(data))

# Safety check
if len(data) < 10:
    raise ValueError("❌ Dataset too small! Add more images.")

# =========================
# 2. PREPROCESS DATA
# =========================

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# =========================
# 3. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# =========================
# 4. LOAD MOBILENET
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model (important)
for layer in base_model.layers:
    layer.trainable = False

# =========================
# 5. CUSTOM HEAD
# =========================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# 6. COMPILE
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 7. TRAIN
# =========================

print("🚀 Training started...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# =========================
# 8. SAVE MODEL (IMPORTANT)
# =========================

save_path = os.path.join(BASE_DIR, "models", "mobilenet_model")
os.makedirs(save_path, exist_ok=True)

model.save(save_path)

print("✅ MobileNet Model Saved Successfully (SavedModel format)")