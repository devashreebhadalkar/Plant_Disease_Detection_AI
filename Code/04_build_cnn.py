import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
train_path = "Dataset/archive (3)/PlantDoc-Dataset/train"
test_path = "Dataset/archive (3)/PlantDoc-Dataset/test"

# Image settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load datasets
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

# Get number of classes
class_names = train_dataset.class_names
num_classes = len(class_names)

# Build CNN model
model = models.Sequential([
    
    # First convolution block
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Output layer
    layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model information
print("\n===== CNN MODEL =====")
model.summary()

print("\nNumber of classes:", num_classes)
print("\nCNN model created successfully!")

