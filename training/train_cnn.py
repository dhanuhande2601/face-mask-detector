import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =====================
# LOAD DATASET
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset")

categories = ["with_mask", "without_mask"]

data = []
labels = []

for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)

# =====================
# CONVERT DATA
# =====================

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print("Total images loaded:", len(data))

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# =====================
# ONE HOT ENCODING
# =====================

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# =====================
# CNN MODEL
# =====================

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# =====================
# COMPILE
# =====================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# TRAIN
# =====================

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)

# =====================
# SAVE MODEL
# =====================

os.makedirs("../models", exist_ok=True)
model.save("../models/cnn_model.h5")

print("CNN Model Trained Successfully!")