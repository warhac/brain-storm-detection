import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

train_dir = "Training"
test_dir = "Testing"

input_shape = (150, 150, 3)
epochs = 10
batch_size = 32

train_images = []
train_labels = []

class_names = sorted(os.listdir(train_dir))

print("Loading Training Data...")

for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(train_dir, class_name)

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)

        img = load_img(img_path, target_size=(150, 150))
        img = img_to_array(img)
        img = img / 255.0

        train_images.append(img)
        train_labels.append(class_index)

train_images = np.array(train_images)
train_labels = to_categorical(train_labels, num_classes=len(class_names))

print("Loading Testing Data...")

test_images = []
test_labels = []

for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(test_dir, class_name)

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)

        img = load_img(img_path, target_size=(150, 150))
        img = img_to_array(img)
        img = img / 255.0

        test_images.append(img)
        test_labels.append(class_index)

test_images = np.array(test_images)
test_labels = to_categorical(test_labels, num_classes=len(class_names))

print("Building Model...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Started...")

model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels)
)

print("Saving Model...")

model.save("brain_tumor_model2.h5")

print("Training Complete! Model Saved Successfully.")
