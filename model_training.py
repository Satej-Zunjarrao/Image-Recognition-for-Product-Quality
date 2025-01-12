"""
model_training.py

This module builds, trains, and fine-tunes a CNN for classifying product images into "defective"
and "non-defective" categories. The model leverages TensorFlow and Keras for implementation.

Author: Satej
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths
TRAIN_DIR = "/path/to/training_data"  # Directory containing training images
VALIDATION_DIR = "/path/to/validation_data"  # Directory containing validation images
MODEL_SAVE_PATH = "/path/to/saved_model/model.h5"  # Path to save the trained model

# Hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.001
EPOCHS = 20

def create_data_generators(train_dir, validation_dir):
    """
    Creates training and validation data generators for loading and augmenting image data.

    Args:
        train_dir (str): Path to the training data directory.
        validation_dir (str): Path to the validation data directory.

    Returns:
        tuple: Training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

def build_model():
    """
    Builds a CNN model using a pre-trained MobileNetV2 as the base.

    Returns:
        tensorflow.keras.Model: Compiled model.
    """
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators(TRAIN_DIR, VALIDATION_DIR)
    
    # Build and compile the model
    print("Building the model...")
    model = build_model()
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Save the model
    print(f"Saving the model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model training complete.")
