import tensorflow as tf

# Dataset paths
train_path = "Dataset/archive (3)/PlantDoc-Dataset/train"
test_path = "Dataset/archive (3)/PlantDoc-Dataset/test"

# Image settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load testing dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = train_dataset.class_names

print("\n===== DATASET PREPROCESSING =====")
print("Number of classes:", len(class_names))
print("Classes:")

for i, class_name in enumerate(class_names):
    print(i, ":", class_name)

print("\nImage size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nDataset loaded successfully!") 
