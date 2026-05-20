# Convolutional neural network (CNN) Use any dataset of plant disease and design a plant 
# disease detection system using CNN. 

# !pip install tensorflow tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(dataset_train, dataset_test), dataset_info = tfds.load(
    "plant_village",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

num_classes = dataset_info.features["label"].num_classes
class_names = dataset_info.features["label"].names

print("Number of classes:", num_classes)
print("Example classes:", class_names[:10])

for image, label in dataset_train.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

IMG_SIZE = 128

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_data = dataset_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = dataset_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

plt.figure(figsize=(8,8))

for i, (image, label) in enumerate(train_data.take(1)):
    for j in range(9):
        plt.subplot(3,3,j+1)
        plt.imshow(image[j])
        plt.title(class_names[label[j]])
        plt.axis("off")

plt.show()

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=3,
    validation_data=test_data
)

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()

for image, label in test_data.take(1):

    prediction = model.predict(image)

    predicted_class = prediction.argmax(axis=1)

    plt.imshow(image[0])
    plt.title("Predicted: " + class_names[predicted_class[0]])
    plt.axis("off")

    break
















# ── STEP 1: Install required libraries ───────────────────────
# Run this in terminal first:
# pip install tensorflow kaggle matplotlib seaborn scikit-learn

# ── STEP 2: Download dataset ──────────────────────────────────
# Run this in terminal:
# pip install kaggle
# Place kaggle.json in C:/Users/YourName/.kaggle/ (Windows)
# kaggle datasets download -d abdallahalidev/plantvillage-dataset
# unzip plantvillage-dataset.zip

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── DATASET PATH ──────────────────────────────────────────────
# Change this to where you unzipped the dataset
DATASET_PATH = "plantvillage dataset/color"        # Update this path
IMG_SIZE     = 128
BATCH_SIZE   = 32

# Check if path exists
if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset path not found!")
    print("Please update DATASET_PATH to your dataset location")
else:
    print("Dataset found at:", DATASET_PATH)
    print("Number of classes:", len(os.listdir(DATASET_PATH)))

# ── a. DATA PRE-PROCESSING ────────────────────────────────────

# ImageDataGenerator handles loading + augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,                                # Normalize 0-1
    validation_split=0.2,                          # 80-20 split
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',                             # Training split
    seed=42
)

# Load test data
test_data = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',                           # Validation split
    seed=42
)

# Class info
class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)
print(f"\nTotal Classes  : {num_classes}")
print(f"Train Samples  : {train_data.samples}")
print(f"Test Samples   : {test_data.samples}")

# Show sample images
plt.figure(figsize=(12, 6))
images, labels = next(train_data)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(class_names[int(labels[i])][:20], fontsize=7)
    plt.axis("off")
plt.suptitle("Sample Plant Disease Images")
plt.tight_layout()
plt.show()

# ── b. DEFINE & TRAIN MODEL ───────────────────────────────────
model = keras.models.Sequential([

    # First Conv Block
    keras.layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(2, 2),

    # Second Conv Block
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # Third Conv Block
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# ── c. EVALUATE RESULTS ───────────────────────────────────────
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

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

# ── CONFUSION MATRIX ──────────────────────────────────────────
print("\nGenerating Confusion Matrix...")

y_true, y_pred = [], []

for i in range(len(test_data)):
    images, labels = next(test_data)
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Top 10 classes only for readability
top_n  = 10
cm     = confusion_matrix(y_true, y_pred)
cm_top = cm[:top_n, :top_n]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[:top_n],
            yticklabels=class_names[:top_n])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Top 10 Classes)")
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report (Top 10 classes):")
print(classification_report(y_true, y_pred,
      target_names=class_names, labels=list(range(top_n))))

# ── SAMPLE PREDICTIONS ────────────────────────────────────────
plt.figure(figsize=(12, 5))
images, labels = next(test_data)
preds          = model.predict(images, verbose=0)
pred_classes   = np.argmax(preds, axis=1)

for j in range(10):
    plt.subplot(2, 5, j+1)
    plt.imshow(images[j])
    actual    = class_names[int(labels[j])]
    predicted = class_names[pred_classes[j]]
    color     = 'green' if actual == predicted else 'red'
    plt.title(f"A:{actual[:12]}\nP:{predicted[:12]}",
              fontsize=6, color=color)
    plt.axis("off")

plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.tight_layout()
plt.show()
