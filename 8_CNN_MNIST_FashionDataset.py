# Use MNIST Fashion Dataset and create a classifier to classify fashion clothing into categories. 
# Using CNN

#Install TensorFlow, NumPy, and Matplotlib required for deep learning and visualization.
#!pip install tensorflow matplotlib numpy

#Import required Python libraries for building the CNN model and handling data.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load the Fashion MNIST dataset which contains training and testing clothing images.
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Training data size:", len(train_images))
print("Test data size:", len(test_images))

print("Shape of training images:", train_images.shape)
print("Shape of test images:", test_images.shape)

print("Training labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)

print("Minimum pixel value:", train_images.min())
print("Maximum pixel value:", train_images.max())

print("Unique labels:", np.unique(train_labels))


import pandas as pd

counts = pd.Series(train_labels).value_counts()
print(counts)

#Create a list that maps numeric labels (0–9) to clothing category names.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Display the size and dimensions of the training and testing data.
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)

#Scale pixel values from 0–255 to 0–1 to improve neural network performance.
train_images = train_images / 255.0
test_images = test_images / 255.0

#Add a channel dimension so the CNN can process the images correctly.
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

#Display a few images from the dataset to understand what the data looks like.
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i].reshape(28,28), )
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.show()

#Display a few images from the dataset to understand what the data looks like.
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i].reshape(28,28), cmap='gray')
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.show()

#Create a Convolutional Neural Network with convolution, pooling, and dense layers.
model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.Flatten(),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(10, activation='softmax')
])

#Configure the model by selecting optimizer, loss function, and accuracy metric.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train the CNN using the training images so it can learn clothing patterns.
history = model.fit(
    train_images,
    train_labels,
    epochs=15,
    validation_split=0.2
)

#Test the trained model using test data to measure its accuracy.
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_acc)

#Use the trained model to predict clothing categories for new images.
predictions = model.predict(test_images)

#Display the image along with its predicted clothing label.
index = 12

plt.imshow(test_images[index].reshape(28,28), cmap='gray')
plt.title("Actual: " + class_names[test_labels[index]])
plt.show()

predicted_label = np.argmax(predictions[index])

print("Predicted:", class_names[predicted_label])

#Visualize training performance using accuracy and loss graphs.
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Model Loss")
plt.show()



















import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── LOAD DATASET ──────────────────────────────────────────────
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Train size :", train_images.shape)
print("Test size  :", test_images.shape)
print("Min pixel  :", train_images.min())
print("Max pixel  :", train_images.max())

# ── a. DATA PRE-PROCESSING ────────────────────────────────────

# Normalize 0-255 → 0-1
train_images = train_images / 255.0
test_images  = test_images  / 255.0

# Add channel dimension for CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images  = test_images.reshape(-1, 28, 28, 1)

# Show sample images                                # Fixed: removed duplicate plot
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.suptitle("Sample Fashion MNIST Images")
plt.tight_layout()
plt.show()

# ── b. DEFINE & TRAIN MODEL ───────────────────────────────────
model = keras.models.Sequential([

    # First Conv Block
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    # Second Conv Block
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # Third Conv Block
    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.Flatten(),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),                      # Fixed: added to reduce overfitting

    # Output Layer
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_images,
    train_labels,
    epochs=15,
    validation_split=0.2
)

# ── c. EVALUATE RESULTS ───────────────────────────────────────
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

# Accuracy & Loss Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ── CONFUSION MATRIX ──────────────────────────────────────────  # Fixed: was missing
predictions    = model.predict(test_images)
y_pred_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Fashion MNIST")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(test_labels, y_pred_classes, target_names=class_names))

# ── SAMPLE PREDICTIONS ────────────────────────────────────────
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    actual    = class_names[test_labels[i]]
    predicted = class_names[y_pred_classes[i]]
    color     = 'green' if actual == predicted else 'red'
    plt.title(f"A:{actual[:8]}\nP:{predicted[:8]}", fontsize=7, color=color)
    plt.axis('off')
plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.tight_layout()
plt.show()

# ── WRONG PREDICTIONS ─────────────────────────────────────────  # Fixed: was missing
wrong_idx = np.where(y_pred_classes != test_labels)[0]
print(f"\nTotal Wrong: {len(wrong_idx)} / {len(test_labels)}")

plt.figure(figsize=(12, 5))
for i, idx in enumerate(wrong_idx[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"A:{class_names[test_labels[idx]][:8]}\n"
              f"P:{class_names[y_pred_classes[idx]][:8]}",
              fontsize=7, color='red')
    plt.axis('off')
plt.suptitle("Wrong Predictions (Actual vs Predicted)")
plt.tight_layout()
plt.show()