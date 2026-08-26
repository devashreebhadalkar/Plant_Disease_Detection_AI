import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# DATASET PATHS
# ==============================

train_path = "Dataset/archive (3)/PlantDoc-Dataset/train"
test_path = "Dataset/archive (3)/PlantDoc-Dataset/test"

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# ==============================
# LOAD DATASET
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)

print("\n===== TRAINING INFORMATION =====")
print("Number of classes:", num_classes)
print("Training images: 2316")
print("Testing images: 236")
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

# ==============================
# BUILD CNN MODEL
# ==============================

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Normalize pixel values
    layers.Rescaling(1./255),

    # Convolution Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# TRAIN MODEL
# ==============================

print("\n===== STARTING CNN TRAINING =====")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# Save training history
import json

with open("Results/07_training_history.json", "w") as file:
    json.dump(history.history, file)

print("\nTraining history saved successfully!")

# ==============================
# SAVE MODEL
# ==============================

model.save("plant_disease_model.keras")

print("\n===== TRAINING COMPLETED =====")
print("Model saved as: plant_disease_model.keras")

