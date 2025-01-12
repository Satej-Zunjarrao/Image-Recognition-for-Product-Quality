"""
feature_extraction.py

This module extracts visual features such as edges, textures, and color distributions
to aid in product defect detection. The features are saved for potential use in visualization
or analysis.

Author: Satej
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define input and output directories
INPUT_DIR = "/path/to/processed_dataset"  # Directory containing processed images
FEATURE_SAVE_DIR = "/path/to/features"  # Directory to save extracted features

def extract_edge_features(image):
    """
    Extracts edge features from an image using the Canny edge detection algorithm.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Edge-detected image.
    """
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    return edges

def extract_texture_features(image):
    """
    Extracts texture features using the Laplacian operator.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with extracted texture features.
    """
    texture = cv2.Laplacian(image, cv2.CV_64F)
    return np.abs(texture)

def extract_color_distribution(image):
    """
    Extracts color distribution (histogram) from the image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        list: Color histograms for each channel.
    """
    histograms = []
    for channel in range(3):  # Assuming RGB image
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        histograms.append(hist)
    return histograms

def save_feature_visualizations(features, filename, output_dir):
    """
    Saves visualizations of extracted features for analysis.

    Args:
        features (dict): Dictionary of extracted features.
        filename (str): Name of the original image file.
        output_dir (str): Directory to save feature visualizations.
    """
    base_name, _ = os.path.splitext(filename)
    feature_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
    for feature_name, feature_image in features.items():
        save_path = os.path.join(feature_dir, f"{feature_name}.png")
        plt.imsave(save_path, feature_image, cmap='gray')

if __name__ == "__main__":
    # Load processed images
    print("Loading processed images...")
    images = []
    filenames = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(INPUT_DIR, filename), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    
    # Extract features from each image
    print("Extracting features...")
    for image, filename in zip(images, filenames):
        edge_features = extract_edge_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        texture_features = extract_texture_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        color_distribution = extract_color_distribution(image)
        
        # Save visualizations of extracted features
        features = {
            "edges": edge_features,
            "texture": texture_features,
        }
        print(f"Saving features for {filename}...")
        save_feature_visualizations(features, filename, FEATURE_SAVE_DIR)
    
    print("Feature extraction complete.")
