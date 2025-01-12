"""
api_integration.py

This module integrates the trained CNN model into an API for real-time product defect detection.
The API uses Flask for deployment and is designed to run on AWS.

Author: Satej
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Define paths
MODEL_PATH = "/path/to/saved_model/optimized_model.h5"  # Path to the optimized model
UPLOAD_FOLDER = "/path/to/uploaded_images"  # Folder to temporarily save uploaded images

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    """
    Preprocesses an input image for model prediction.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize to 0-1 range
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting if a product is defective or non-defective.

    Returns:
        json: Prediction result.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Preprocess the image
    img = preprocess_image(file_path)
    
    # Make a prediction
    prediction = model.predict(img)
    os.remove(file_path)  # Clean up the uploaded file
    
    # Interpret the prediction
    result = "Defective" if prediction[0][0] > 0.5 else "Non-Defective"
    return jsonify({"prediction": result, "confidence": float(prediction[0][0])})

if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000)
