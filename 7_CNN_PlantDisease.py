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
















import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── LOAD DATASET ──────────────────────────────────────────────
(dataset_train, dataset_test), dataset_info = tfds.load(
    "plant_village",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

num_classes = dataset_info.features["label"].num_classes
class_names = dataset_info.features["label"].names

print("Number of classes:", num_classes)
print("Example classes  :", class_names[:5])

# ── a. DATA PRE-PROCESSING ────────────────────────────────────
IMG_SIZE = 128

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0                            # Normalize 0-1
    return image, label

train_data = dataset_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_data  = dataset_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Show sample images
plt.figure(figsize=(10, 8))
for i, (image, label) in enumerate(train_data.take(1)):
    for j in range(9):
        plt.subplot(3, 3, j+1)
        plt.imshow(image[j])
        plt.title(class_names[label[j]], fontsize=7)
        plt.axis("off")
plt.suptitle("Sample Plant Disease Images")
plt.tight_layout()
plt.show()

# ── b. DEFINE & TRAIN MODEL ───────────────────────────────────
model = tf.keras.models.Sequential([

    # First Conv Block
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second Conv Block
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third Conv Block
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),                   # Fixed: added to reduce overfitting

    tf.keras.layers.Dense(num_classes, activation='softmax')
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
    epochs=3,
    validation_data=test_data
)

# ── c. EVALUATE RESULTS ───────────────────────────────────────
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# Accuracy Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss Plot                                         # Fixed: was missing
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ── CONFUSION MATRIX ──────────────────────────────────────────  # Fixed: was missing
print("\nGenerating Confusion Matrix...")

y_true, y_pred = [], []

for image, label in test_data:
    preds = model.predict(image, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(label.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Plot Confusion Matrix (top 10 classes only for readability)
cm         = confusion_matrix(y_true, y_pred)
top_n      = 10
cm_top     = cm[:top_n, :top_n]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[:top_n],
            yticklabels=class_names[:top_n])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Top 10 Classes)")
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report (first 10 classes):")
print(classification_report(y_true, y_pred,
      target_names=class_names, labels=list(range(top_n))))

# ── SAMPLE PREDICTIONS ────────────────────────────────────────
print("\n── Sample Predictions ──")
plt.figure(figsize=(12, 5))

for image, label in test_data.take(1):
    preds           = model.predict(image, verbose=0)
    predicted_class = np.argmax(preds, axis=1)

    for j in range(10):
        plt.subplot(2, 5, j+1)
        plt.imshow(image[j])
        actual    = class_names[label[j]]
        predicted = class_names[predicted_class[j]]
        color     = 'green' if actual == predicted else 'red'  # Green=correct Red=wrong
        plt.title(f"A: {actual[:10]}\nP: {predicted[:10]}",
                  fontsize=6, color=color)
        plt.axis("off")

plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.tight_layout()
plt.show()