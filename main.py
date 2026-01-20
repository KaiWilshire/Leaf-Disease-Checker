# ==========================================
# PLANT LEAF DISEASE CLASSIFICATION SYSTEM
# ==========================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.feature import hog

# ==========================================
# IMAGE PREPROCESSING
# ==========================================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Gaussian filtering (noise removal)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# ==========================================
# HARRIS CORNER DETECTION (DISEASE SPOTS)
# ==========================================
def harris_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    img[corners > 0.01 * corners.max()] = [255, 0, 0]

    return img


# ==========================================
# HOG FEATURES (LEAF SHAPE & VEINS)
# ==========================================
def hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features, hog_img = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True
    )
    return hog_img


# ==========================================
# CNN MODEL
# ==========================================
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Healthy / Diseased
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================
# TRAINING PIPELINE
# ==========================================
def train_model():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        image_size=(224, 224),
        batch_size=16,
        label_mode="binary"
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/test",
        image_size=(224, 224),
        batch_size=16,
        label_mode="binary"
    )

    model = build_cnn()
    model.fit(train_data, epochs=10, validation_data=test_data)

    model.save("leaf_disease_model.h5")
    return model


# ==========================================
# PREDICTION
# ==========================================
def predict_leaf(model, img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("Leaf Status: Diseased")
    else:
        print("Leaf Status: Healthy")


# ==========================================
# VISUALIZATION (DEMO)
# ==========================================
def visualize_features(img_path):
    img = preprocess_image(img_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Harris Corners")
    plt.imshow(harris_corners(img.copy()))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("HOG Features")
    plt.imshow(hog_features(img), cmap="gray")
    plt.axis("off")

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Train the model (run once)
    model = train_model()

    # Test on a sample image
    sample_image = "sample_leaf.jpg"  # change this
    visualize_features(sample_image)
    predict_leaf(model, sample_image)
