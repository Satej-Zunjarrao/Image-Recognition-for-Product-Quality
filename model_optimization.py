"""
model_optimization.py

This module applies optimization techniques to improve the CNN's performance by
reducing overfitting and enhancing training efficiency. Techniques include learning rate
scheduling and dropout regularization.

Author: Satej
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Define paths
MODEL_PATH = "/path/to/saved_model/model.h5"  # Path to the pre-trained model
OPTIMIZED_MODEL_PATH = "/path/to/saved_model/optimized_model.h5"  # Path to save the optimized model

# Hyperparameters
INITIAL_LEARNING_RATE = 0.001
DROP_RATE = 0.5

def apply_dropout(model):
    """
    Adds dropout layers to reduce overfitting.

    Args:
        model (tensorflow.keras.Model): Pre-trained model.

    Returns:
        tensorflow.keras.Model: Model with dropout layers added.
    """
    new_model = tf.keras.models.Sequential()
    for layer in model.layers:
        new_model.add(layer)
        if isinstance(layer, tf.keras.layers.Dense):
            new_model.add(tf.keras.layers.Dropout(DROP_RATE))
    return new_model

def lr_scheduler(epoch, lr):
    """
    Custom learning rate scheduler.

    Args:
        epoch (int): Current epoch.
        lr (float): Current learning rate.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch < 5:
        return lr
    return lr * tf.math.exp(-0.1)

def optimize_model(model_path, optimized_model_path):
    """
    Optimizes a pre-trained model by applying dropout and learning rate scheduling.

    Args:
        model_path (str): Path to the pre-trained model.
        optimized_model_path (str): Path to save the optimized model.
    """
    # Load the pre-trained model
    print("Loading model...")
    model = load_model(model_path)
    
    # Apply dropout
    print("Applying dropout...")
    optimized_model = apply_dropout(model)
    optimized_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    
    # Define callbacks
    callbacks = [
        LearningRateScheduler(lr_scheduler),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    # Training the model again with callbacks
    print("Training optimized model...")
    # Assuming training data and validation data generators are already created
    train_generator = ...  # Replace with your training generator
    validation_generator = ...  # Replace with your validation generator
    
    optimized_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=callbacks
    )
    
    # Save the optimized model
    print(f"Saving optimized model to {optimized_model_path}...")
    optimized_model.save(optimized_model_path)
    print("Model optimization complete.")

if __name__ == "__main__":
    optimize_model(MODEL_PATH, OPTIMIZED_MODEL_PATH)
