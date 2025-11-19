import tensorflow as tf
# --- FIX FOR GPU/PTXAS ERROR: Force TensorFlow to use the CPU ---
try:
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    print("FIX APPLIED: GPU devices disabled to bypass ptxas/nvlink compilation issue. Running on CPU.")
except Exception as e:
    print(f"Could not disable GPU devices: {e}")
# ----------------------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --- 0. Setup and Constants ---
print("\n--- 0. Setup and Constants ---")
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define Image Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 15
EPOCHS = 80
DATA_DIR = 'Waste_Classification_FLAT' 

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Data Generators with Augmentation and Preprocessing ---
print("\n--- 1. Setting up Data Generators ---")
# Rescale the image pixel values from 0-255 to 0-1 and apply augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest',
    validation_split=0.2 # Use 20% of the data for validation/testing
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    DATA_DIR, 
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Specify this is the training set
)

# Load Validation Data
validation_generator = datagen.flow_from_directory(
    DATA_DIR, 
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Specify this is the validation set
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Total Training Images: {train_generator.samples}")
print(f"Total Validation Images: {validation_generator.samples}")
print(f"Number of Classes: {NUM_CLASSES}")


# --- 2. Build the Transfer Learning Model (ResNet50) ---
print("\n--- 2. Building ResNet50 Transfer Learning Model ---")
# Load the pre-trained ResNet50 base model
base_model = ResNet50(
    weights='imagenet', 
    include_top=False, # Exclude the original classification head
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3) # 3 for RGB channels
)

# Freeze the base layers to prevent their weights from being updated
for layer in base_model.layers:
    layer.trainable = False

# Create the new model structure (Classification Head)
model = Sequential([
    base_model, # Add the pre-trained ResNet50 base
    GlobalAveragePooling2D(), # Efficiently reduces the feature map size
    Dense(512, activation='relu'), # Intermediate, dense layer
    Dropout(0.5), # Dropout for regularization/preventing overfitting
    Dense(NUM_CLASSES, activation='softmax') # Final output layer with 30 nodes (one for each class)
])

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Low learning rate for transfer learning
    loss='categorical_crossentropy', # Required for multi-class classification with one-hot encoding
    metrics=['accuracy']
)

print("Model Structure Summary:")
model.summary()


# --- 3. Train the Model ---
print(f"\n--- 3. Starting Training for {EPOCHS} epochs... ---")

# Define the number of steps per epoch
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1 # Show the progress bar and metrics
)

print("Training complete.")


# --- 4. Plot Training Results ---
print("\n--- 4. Plotting Training and Validation Metrics ---")
try:
    # Retrieve the training metrics stored in the 'history' object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout() # Adjusts plot to fit titles nicely
    plt.savefig('training_metrics.png') # Save plot to file

except NameError:
    print("Could not plot history. This is likely because training was interrupted.")
except AttributeError:
    print("Could not plot history. The 'history' object may be incomplete.")


# --- 5. Save the Model and Class Indices ---
print("\n--- 5. Saving Model and Class Indices ---")

MODEL_FILENAME = 'waste_classifier_resnet50.h5'
INDICES_FILENAME = 'class_indices.json'

try:
    # Save the model to an HDF5 file
    model.save(MODEL_FILENAME) 
    print(f"Model saved successfully as {MODEL_FILENAME}")

    # Save the class indices mapping for label display
    with open(INDICES_FILENAME, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Class indices saved successfully as {INDICES_FILENAME}")

except Exception as e:
    print(f"An error occurred during saving: {e}")