
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import mobilenet_v2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

MODEL_PATH = "plant_disease_model.keras"
TEST_PATH = r"Dataset\archive (3)\PlantDoc-Dataset\test"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

RESULTS_PATH = "Results"
CLASS_NAMES_PATH = os.path.join(RESULTS_PATH, "class_names.json")


# ============================================================
# 2. CREATE RESULTS FOLDER
# ============================================================

os.makedirs(RESULTS_PATH, exist_ok=True)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

print("\n===== CHECKING REQUIRED FILES =====")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(
        f"Test dataset folder not found: {TEST_PATH}"
    )

if not os.path.isfile(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


# ============================================================
# 4. LOAD CLASS NAMES
# ============================================================

print("\n===== LOADING CLASS NAMES =====")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError(
        "class_names.json must contain a non-empty list."
    )

if len(class_names) != len(set(class_names)):
    raise ValueError(
        "Duplicate class names found in class_names.json."
    )

print(f"Number of classes: {len(class_names)}")


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\n===== LOADING TRAINED MODEL =====")

model = tf.keras.models.load_model(MODEL_PATH)

model_output_classes = int(model.output_shape[-1])

if model_output_classes != len(class_names):
    raise ValueError(
        f"Model outputs {model_output_classes} classes, but "
        f"class_names.json contains {len(class_names)} classes."
    )

print("Model loaded successfully!")
print(f"Model output classes: {model_output_classes}")


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

print("\n===== LOADING TEST DATASET =====")

# Use the saved training class order rather than allowing
# the loader to create a new class-to-index mapping.
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names,
    label_mode="int",
)

number_of_test_images = len(test_dataset.file_paths)

print(f"Test images: {number_of_test_images}")

if number_of_test_images == 0:
    raise ValueError("The test dataset contains no images.")


# ============================================================
# 7. APPLY MOBILENETV2 PREPROCESSING
# ============================================================

print("\n===== PREPROCESSING TEST IMAGES =====")


def preprocess_test_images(images, labels):
    """
    Apply the same MobileNetV2 preprocessing used by the model.
    """

    images = tf.cast(images, tf.float32)

    images = mobilenet_v2.preprocess_input(images)

    return images, labels


test_dataset = test_dataset.map(
    preprocess_test_images,
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

print("\n===== MODEL EVALUATION =====")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1,
)

print("\n===== EVALUATION RESULTS =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================

print("\n===== GENERATING PREDICTIONS =====")

true_labels = []
predicted_labels = []

for images, labels in test_dataset:
    probabilities = model.predict(images, verbose=0)

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels.extend(labels.numpy().tolist())
    predicted_labels.extend(predicted_classes.tolist())

true_labels = np.asarray(true_labels, dtype=np.int64)
predicted_labels = np.asarray(predicted_labels, dtype=np.int64)

if len(true_labels) != number_of_test_images:
    raise RuntimeError(
        "The number of evaluated images does not match "
        "the number of test images."
    )


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

class_indices = list(range(len(class_names)))

report = classification_report(
    true_labels,
    predicted_labels,
    labels=class_indices,
    target_names=class_names,
    zero_division=0,
)

print(report)

report_path = os.path.join(
    RESULTS_PATH,
    "08_classification_report.txt",
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Plant Disease Classification Report\n")
    file.write("===================================\n\n")
    file.write(f"Test images: {number_of_test_images}\n")
    file.write(f"Number of classes: {len(class_names)}\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
    file.write(report)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n===== GENERATING CONFUSION MATRIX =====")

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=class_indices,
)

fig, ax = plt.subplots(figsize=(18, 15))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=False,
)

ax.set_title(
    "Plant Disease Classification - Confusion Matrix"
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_PATH,
    "08_confusion_matrix.png",
)

plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 12. SAVE TEST RESULTS AS JSON
# ============================================================

results_path = os.path.join(
    RESULTS_PATH,
    "08_test_results.json",
)

results = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_accuracy_percentage": float(test_accuracy * 100),
    "number_of_test_images": int(number_of_test_images),
    "number_of_classes": int(len(class_names)),
    "class_names": class_names,
}

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n===== FILES SAVED =====")
print(f"Classification report: {report_path}")
print(f"Confusion matrix: {confusion_matrix_path}")
print(f"Test results: {results_path}")

print("\nModel evaluation completed successfully!")


# ============================================================
# PURPOSE OF THIS PROGRAM
# ============================================================
#
# Evaluates the trained MobileNetV2 plant-image classifier on
# the separate PlantDoc test dataset.
#
# It performs the following tasks:
# 1. Loads the trained model and saved class-name mapping.
# 2. Loads test images using the saved class order.
# 3. Applies MobileNetV2 image preprocessing.
# 4. Calculates test loss and test accuracy.
# 5. Generates a classification report and confusion matrix.
# 6. Saves the evaluation results in the Results folder.
#
# IMPORTANT LIMITATIONS:
# The class mapping must match the mapping used during training.
# The evaluation measures performance on this test dataset only.
# Incorrect predictions are possible, including on real-world images.
# Test accuracy does not guarantee correct individual predictions.
#
# ============================================================
