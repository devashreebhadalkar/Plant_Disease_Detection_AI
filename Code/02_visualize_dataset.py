import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
train_path = "Dataset/archive (3)/PlantDoc-Dataset/train"

# Get all class folders
classes = [
    folder for folder in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, folder))
]

# Select 9 random classes
selected_classes = random.sample(classes, 9)

# Create figure
plt.figure(figsize=(12, 10))

for i, class_name in enumerate(selected_classes):
    class_path = os.path.join(train_path, class_name)

    # Get image files
    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Select one random image
    image_name = random.choice(image_files)
    image_path = os.path.join(class_path, image_name)

    # Open image
    image = Image.open(image_path)

    # Display image
    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_name, fontsize=9)
    plt.axis("off")

plt.suptitle("Sample Images from PlantDoc Dataset", fontsize=16)
plt.tight_layout()

# Save the figure
plt.savefig("Results/sample_leaf_images.png", dpi=300)

plt.show()
