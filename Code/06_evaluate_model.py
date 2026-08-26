import tensorflow as tf

# ==============================
# PATHS
# ==============================

test_path = "Dataset/archive (3)/PlantDoc-Dataset/test"
model_path = "plant_disease_model.keras"

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# ==============================
# LOAD TEST DATASET
# ==============================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model(model_path)

# ==============================
# EVALUATE MODEL
# ==============================

print("\n===== MODEL EVALUATION =====")

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n===== FINAL RESULTS =====")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)

