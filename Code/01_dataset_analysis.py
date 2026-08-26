import os

# Dataset paths
train_path = "Dataset/archive (3)/PlantDoc-Dataset/train"
test_path = "Dataset/archive (3)/PlantDoc-Dataset/test"

# Get class names
train_classes = os.listdir(train_path)
test_classes = os.listdir(test_path)

print("===== PLANT DISEASE DATASET ANALYSIS =====")

print("\nNumber of training classes:", len(train_classes))
print("Number of testing classes:", len(test_classes))

print("\nTraining Classes:")
for class_name in train_classes:
    print(class_name)

# Count training images
train_images = 0

for class_name in train_classes:
    class_path = os.path.join(train_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        train_images += len(images)

# Count testing images
test_images = 0

for class_name in test_classes:
    class_path = os.path.join(test_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        test_images += len(images)

print("\nTotal training images:", train_images)
print("Total testing images:", test_images)
print("Total images:", train_images + test_images)

