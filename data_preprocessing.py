"""
data_preprocessing.py

This module handles image loading, preprocessing, and standardization for defect detection.
The goal is to prepare the dataset for machine learning by cleaning and standardizing images.

Author: Satej
"""

import cv2
import pandas as pd
import numpy as np
import os

# Define the directory containing the images
IMAGE_DIR = "/path/to/dataset"  # Replace with your dataset directory
OUTPUT_DIR = "/path/to/processed_dataset"  # Directory for saving processed images

def load_images(image_dir):
    """
    Loads images from the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        list: A list of tuples containing the image and its filename.
    """
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is not None:
                images.append((img, filename))
    return images

def preprocess_image(image):
    """
    Preprocesses an image by resizing, enhancing contrast, and removing noise.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Processed image.
    """
    # Resize the image to a standard size (e.g., 224x224)
    resized = cv2.resize(image, (224, 224))
    
    # Convert the image to grayscale for consistent processing
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Normalize the image to a 0-1 range
    normalized = denoised / 255.0
    
    return normalized

def save_processed_images(images, output_dir):
    """
    Saves processed images to the specified directory.

    Args:
        images (list): List of processed images and their filenames.
        output_dir (str): Directory to save processed images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img, filename in images:
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))  # Convert back to 0-255 range before saving

if __name__ == "__main__":
    # Load images from the dataset
    print("Loading images...")
    raw_images = load_images(IMAGE_DIR)
    
    # Preprocess each image
    print("Preprocessing images...")
    processed_images = [(preprocess_image(img), name) for img, name in raw_images]
    
    # Save the processed images
    print(f"Saving processed images to {OUTPUT_DIR}...")
    save_processed_images(processed_images, OUTPUT_DIR)
    print("Preprocessing complete.")
