import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 3
DATASET_DIR = "dataset"

# -----------------------------
# DATA
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -----------------------------
# CNN-A (SMALL)
# -----------------------------
cnn_A = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn_A.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history_A = cnn_A.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# -----------------------------
# CNN-B (DEEPER)
# -----------------------------
cnn_B = Sequential([
    Conv2D(32, (5,5), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn_B.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history_B = cnn_B.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# -----------------------------
# ACCURACY PLOTS WITH VALUES
# -----------------------------
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(10,4))

# Validation Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, history_A.history["val_accuracy"], marker="o", label="CNN-A")
plt.plot(epochs_range, history_B.history["val_accuracy"], marker="o", label="CNN-B")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

for i, v in enumerate(history_A.history["val_accuracy"]):
    plt.text(i+1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(history_B.history["val_accuracy"]):
    plt.text(i+1, v, f"{v:.2f}", ha="center", va="top", fontsize=9)

# Training Accuracy
plt.subplot(1,2,2)
plt.plot(epochs_range, history_A.history["accuracy"], marker="o", label="CNN-A")
plt.plot(epochs_range, history_B.history["accuracy"], marker="o", label="CNN-B")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

for i, v in enumerate(history_A.history["accuracy"]):
    plt.text(i+1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(history_B.history["accuracy"]):
    plt.text(i+1, v, f"{v:.2f}", ha="center", va="top", fontsize=9)

plt.tight_layout()
plt.show()

# -----------------------------
# COMPARISON TABLE
# -----------------------------
comparison = pd.DataFrame({
    "Model": ["CNN-A", "CNN-B"],
    "Conv Layers": [2, 3],
    "Kernel Sizes": ["3x3", "5x5 + 3x3"],
    "Epochs": [EPOCHS, EPOCHS],
    "Final Train Accuracy": [
        round(history_A.history["accuracy"][-1], 4),
        round(history_B.history["accuracy"][-1], 4)
    ],
    "Final Val Accuracy": [
        round(history_A.history["val_accuracy"][-1], 4),
        round(history_B.history["val_accuracy"][-1], 4)
    ],
    "Parameters": [
        cnn_A.count_params(),
        cnn_B.count_params()
    ]
})

print("\nCNN Comparison Table:")
print(comparison)
