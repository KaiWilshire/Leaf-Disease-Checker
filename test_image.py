import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "leaf_disease_model.h5"
IMG_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# -----------------------------
# LOAD TEST IMAGE
# -----------------------------
# CHANGE THIS PATH to any image you want to test
IMAGE_PATH = "dataset/test/healthy/example.jpg"
# or: dataset/test/diseased/example.jpg

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, IMG_SIZE)
img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# PREDICT
# -----------------------------
prediction = model.predict(img_array)[0][0]

if prediction < 0.5:
    label = "DISEASED"
    confidence = (1 - prediction) * 100
else:
    label = "HEALTHY"
    confidence = prediction * 100

# -----------------------------
# DISPLAY RESULT
# -----------------------------
plt.imshow(img)
plt.title(f"Prediction: {label} ({confidence:.2f}%)")
plt.axis("off")
plt.show()
