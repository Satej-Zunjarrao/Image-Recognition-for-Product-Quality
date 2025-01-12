# Image-Recognition-for-Product-Quality
Trained a deep learning model to detect defects in manufacturing products from images.

## Overview
The **Image-Recognition-for-Product-Quality** is a Python-based solution designed to automate the detection of defects in manufacturing lines. By leveraging advanced computer vision techniques and deep learning models, the system ensures high-quality production standards, reduces manual inspection time, and minimizes defective products reaching customers.

This project includes a modular and scalable pipeline for data preprocessing, augmentation, model training, optimization, integration, visualization, and reporting.

---

## Key Features
- **Data Preprocessing**: Standardizes images by resizing, noise removal, and contrast enhancement.
- **Data Augmentation**: Expands training datasets with techniques like flipping, rotation, scaling, and brightness adjustments.
- **Model Training**: Trains convolutional neural networks (CNNs) using TensorFlow and Keras for defect detection.
- **Feature Extraction**: Extracts visual features (edges, textures, color distributions) for enhanced detection.
- **Model Optimization**: Improves model accuracy with learning rate scheduling and dropout techniques.
- **Integration**: Deploys the model on AWS via an API for real-time defect detection.
- **Visualization and Reporting**: Generates dashboards and reports for defect rates, performance metrics, and trends.

---

## Directory Structure

plaintext
project/
│
├── data_preprocessing.py         # Prepares and standardizes images for training
├── data_augmentation.py          # Applies augmentation techniques to expand dataset
├── model_training.py             # Builds and trains CNNs for defect classification
├── feature_extraction.py         # Extracts visual features like edges and textures
├── model_optimization.py         # Optimizes the model for accuracy and efficiency
├── api_integration.py            # Deploys the model using Flask for real-time detection
├── visualization_and_reporting.py# Creates dashboards and reports for monitoring
├── README.md                     # Project documentation

---

# Modules

## 1. data_preprocessing.py
- Loads and preprocesses images (resizing, noise removal, and contrast enhancement).
- Outputs standardized images for training and augmentation.

## 2. data_augmentation.py
- Expands the dataset using flipping, rotation, scaling, and brightness adjustments.
- Saves augmented images for improved model robustness.

## 3. model_training.py
- Builds and trains convolutional neural networks (CNNs) using TensorFlow and Keras.
- Includes transfer learning with pre-trained models like MobileNetV2.

## 4. feature_extraction.py
- Extracts visual features like edges (Canny), textures (Laplacian), and color distributions.
- Outputs features for analysis or debugging.

## 5. model_optimization.py
- Adds dropout layers to reduce overfitting.
- Implements learning rate scheduling and early stopping to enhance model performance.

## 6. api_integration.py
- Deploys the trained model as a REST API using Flask.
- Handles real-time defect detection from uploaded product images.

## 7. visualization_and_reporting.py
- Generates visualizations such as confusion matrices, defect trends, and performance metrics.
- Combines visualizations into a PDF report for stakeholder review.

---

# Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com

