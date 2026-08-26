import tensorflow as tf

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "plant_disease_model.keras"
TEST_PATH = "Dataset/archive (3)/PlantDoc-Dataset/test"

# -----------------------------
# Image settings
# -----------------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# -----------------------------
# Load trained model
# -----------------------------
print("\n===== LOADING TRAINED MODEL =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------
# Load test dataset
# -----------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Evaluate model
# -----------------------------
print("\n===== MODEL EVALUATION =====")

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print("\n===== EVALUATION RESULTS =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nModel evaluation completed successfully!")

