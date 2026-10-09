import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PROJECT SETTINGS
# ============================================================

TRAIN_PATH = r"Dataset\archive (3)\PlantDoc-Dataset\train"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_RATIO = 0.20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs("Results", exist_ok=True)


print("\n===== PLANT DISEASE DETECTION =====")
print("===== TRANSFER LEARNING USING MOBILENETV2 =====")


# ============================================================
# FIND CLASS NAMES
# ============================================================

class_names = sorted(
    [
        folder
        for folder in os.listdir(TRAIN_PATH)
        if os.path.isdir(os.path.join(TRAIN_PATH, folder))
    ]
)

NUM_CLASSES = len(class_names)

print("\n===== CLASS INFORMATION =====")
print("Number of classes:", NUM_CLASSES)

for i, class_name in enumerate(class_names):
    print(i, ":", class_name)


# ============================================================
# CREATE CLASS-BALANCED TRAIN / VALIDATION SPLIT
# ============================================================

print("\n===== CREATING TRAIN / VALIDATION SPLIT =====")

train_paths = []
train_labels = []

validation_paths = []
validation_labels = []

for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(TRAIN_PATH, class_name)

    image_files = [
        os.path.join(class_folder, file)
        for file in os.listdir(class_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - VALIDATION_RATIO))

    class_train = image_files[:split_index]
    class_validation = image_files[split_index:]

    train_paths.extend(class_train)
    train_labels.extend([class_index] * len(class_train))

    validation_paths.extend(class_validation)
    validation_labels.extend([class_index] * len(class_validation))

    print(
        f"{class_name}: "
        f"Train = {len(class_train)}, "
        f"Validation = {len(class_validation)}"
    )


print("\nTotal training images:", len(train_paths))
print("Total validation images:", len(validation_paths))


# ============================================================
# CREATE TENSORFLOW DATASETS
# ============================================================

def load_and_preprocess_image(file_path, label):

    image = tf.io.read_file(file_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    image = preprocess_input(image)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)

train_dataset = train_dataset.shuffle(
    len(train_paths),
    seed=SEED
)

train_dataset = train_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\n===== LOADING MOBILENETV2 =====")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

print("MobileNetV2 loaded successfully!")
print("Pretrained ImageNet layers are frozen.")


# ============================================================
# BUILD TRANSFER LEARNING MODEL
# ============================================================

inputs = layers.Input(
    shape=(160, 160, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n===== MODEL SUMMARY =====")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n===== STARTING TRANSFER LEARNING =====")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    "Results/07_training_history.json",
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    "Results/class_names.json",
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "plant_disease_model.keras"
)


print("\n===== TRAINING COMPLETED =====")

print(
    "Best validation accuracy:",
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    "Best validation loss:",
    f"{min(history.history['val_loss']):.4f}"
)

print("\nSaved files:")
print("Results/07_training_history.json")
print("Results/class_names.json")
print("plant_disease_model.keras")


# ============================================================
# PURPOSE OF THIS PROGRAM
# ============================================================
#
# This program trains a Plant Disease Detection model using
# transfer learning with the pretrained MobileNetV2 network.
#
# The PlantDoc training dataset contains 27 plant/leaf classes.
# The training data is divided into a class-balanced training
# set and validation set.
#
# MobileNetV2 provides pretrained visual features learned from
# the ImageNet dataset. Its pretrained layers are initially
# frozen so that the new classification layers can learn the
# PlantDoc classes.
#
# Data augmentation is used to improve generalization by
# creating small variations of training images.
#
# The program produces:
# 1. A trained plant disease classification model.
# 2. Training and validation history.
# 3. The list of class names.
#
# IMPORTANT LIMITATION:
# The model can only classify images according to the 27
# classes present in the PlantDoc dataset. Its predictions
# depend on the quality and similarity of the input image
# compared with the training data.
#
# ============================================================

