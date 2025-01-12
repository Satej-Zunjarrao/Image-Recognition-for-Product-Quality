"""
data_augmentation.py

This module applies data augmentation techniques to expand the training dataset.
Augmentation improves the model's robustness by artificially increasing dataset diversity.

Author: Satej
"""

import cv2
import numpy as np
import os

# Define the directories for input and augmented images
INPUT_DIR = "/path/to/processed_dataset"  # Replace with the preprocessed dataset directory
AUGMENTED_DIR = "/path/to/augmented_dataset"  # Directory to save augmented images

def augment_image(image):
    """
    Applies a series of augmentation techniques to the given image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        list: A list of augmented versions of the input image.
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Flip the image horizontally
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Rotate the image by 15 degrees
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    augmented_images.append(rotated)
    
    # Scale the image (zoom in)
    scaled = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    scaled_cropped = scaled[:rows, :cols]  # Crop back to original size
    augmented_images.append(scaled_cropped)
    
    # Adjust brightness (increase)
    brightness_increased = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(brightness_increased)
    
    return augmented_images

def save_augmented_images(images, filenames, output_dir):
    """
    Saves augmented images to the specified directory.

    Args:
        images (list): List of augmented images for each input image.
        filenames (list): Corresponding filenames of the input images.
        output_dir (str): Directory to save augmented images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for original_filename, augmented_list in zip(filenames, images):
        base_name, ext = os.path.splitext(original_filename)
        for i, augmented in enumerate(augmented_list):
            save_path = os.path.join(output_dir, f"{base_name}_aug_{i}{ext}")
            cv2.imwrite(save_path, (augmented * 255).astype(np.uint8))  # Convert back to 0-255 range before saving

if __name__ == "__main__":
    # Load preprocessed images
    print("Loading images for augmentation...")
    images = []
    filenames = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(INPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    
    # Apply augmentations to each image
    print("Applying augmentations...")
    augmented_images = [augment_image(img) for img in images]
    
    # Save the augmented images
    print(f"Saving augmented images to {AUGMENTED_DIR}...")
    save_augmented_images(augmented_images, filenames, AUGMENTED_DIR)
    print("Data augmentation complete.")
