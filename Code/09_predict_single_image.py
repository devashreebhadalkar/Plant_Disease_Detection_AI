
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = os.path.join("Results", "class_names.json")
IMAGE_SIZE = (160, 160)


# ============================================================
# 2. LOAD MODEL AND CLASS NAMES
# ============================================================

print("Loading trained MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Model loaded successfully!")
print(f"Number of classes: {len(class_names)}")


# ============================================================
# 3. PREDICT ONE IMAGE
# ============================================================

def predict_image(image_path):

    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    # Load image and resize it to the model's input size
    image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert image to a numerical array
    image_array = img_to_array(image)

    # Add batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Apply the SAME preprocessing used during training
    image_array = mobilenet_v2.preprocess_input(image_array)

    # Get model predictions
    probabilities = model.predict(image_array, verbose=0)[0]

    # Find the class with the highest predicted probability
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    print("\n========== PREDICTION RESULT ==========")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Model confidence: {confidence:.2f}%")
    print("=======================================\n")

    return predicted_class, confidence


# ============================================================
# 4. RUN FROM THE COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict the class of a plant leaf image."
    )

    parser.add_argument(
        "image_path",
        help="Path to the plant leaf image"
    )

    args = parser.parse_args()

    predict_image(args.image_path)


# ============================================================
# PURPOSE OF THIS PROGRAM
# ============================================================
#
# This program loads the trained MobileNetV2 model and predicts
# the class of one plant image.
#
# It resizes the image to 160 x 160 pixels and applies the same
# MobileNetV2 preprocessing used during training.
#
# It displays the predicted class and the model's highest
# predicted probability.
#
# IMPORTANT LIMITATION:
# The model was evaluated at 44.92% accuracy on the separate
# 236-image PlantDoc test dataset. Predictions may be incorrect.
# The displayed confidence is not a guarantee of correctness.
# This program is for educational demonstration, not a
# substitute for expert plant-disease diagnosis.
#
# ============================================================
