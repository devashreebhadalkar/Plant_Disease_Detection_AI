import json
import matplotlib.pyplot as plt

# Load training history
with open("Results/07_training_history.json", "r") as file:
    history = json.load(file)

epochs = range(1, len(history["accuracy"]) + 1)

# -----------------------------
# Training and Validation Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("Results/07_training_accuracy.png")
plt.show()


# -----------------------------
# Training and Validation Loss
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, history["loss"], marker="o", label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("Results/07_training_loss.png")
plt.show()

print("\n===== TRAINING GRAPHS =====")
print("Training accuracy graph saved successfully!")
print("Training loss graph saved successfully!")

