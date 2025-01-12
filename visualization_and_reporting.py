"""
visualization_and_reporting.py

This module creates visualizations and reports for monitoring defect detection results, 
model performance metrics, and production trends. It generates dashboards and exports 
automated reports as files.

Author: Satej
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define paths
RESULTS_FILE = "/path/to/results.csv"  # CSV file containing predictions and actual labels
VISUALIZATION_OUTPUT_DIR = "/path/to/visualizations"  # Directory to save visualizations
REPORT_FILE = "/path/to/report.pdf"  # Path to save the generated report

def load_results(file_path):
    """
    Loads model predictions and ground truth labels from a CSV file.

    Args:
        file_path (str): Path to the results CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    return pd.read_csv(file_path)

def plot_confusion_matrix(results, output_dir):
    """
    Plots the confusion matrix for the defect detection model.

    Args:
        results (pandas.DataFrame): DataFrame containing predictions and actual labels.
        output_dir (str): Directory to save the confusion matrix plot.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_true = results['Actual']
    y_pred = (results['Prediction'] > 0.5).astype(int)  # Convert probabilities to binary labels
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Defective', 'Defective'])
    disp.plot(cmap='Blues', values_format='d')
    
    # Save the plot
    plt.title("Confusion Matrix")
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def plot_performance_metrics(results, output_dir):
    """
    Plots performance metrics over time.

    Args:
        results (pandas.DataFrame): DataFrame containing timestamps and accuracy metrics.
        output_dir (str): Directory to save the performance metrics plot.
    """
    if 'Timestamp' in results.columns and 'Accuracy' in results.columns:
        results['Timestamp'] = pd.to_datetime(results['Timestamp'])
        results = results.sort_values(by='Timestamp')
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['Timestamp'], results['Accuracy'], marker='o', label='Accuracy')
        plt.title("Model Accuracy Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(output_dir, "performance_metrics.png")
        plt.savefig(output_path)
        print(f"Performance metrics saved to {output_path}")
        plt.close()
    else:
        print("Timestamp and/or Accuracy columns are missing in the results file.")

def plot_defect_trends(results, output_dir):
    """
    Plots the trend of defective vs non-defective predictions over time.

    Args:
        results (pandas.DataFrame): DataFrame containing timestamps and predictions.
        output_dir (str): Directory to save the defect trends plot.
    """
    if 'Timestamp' in results.columns and 'Prediction' in results.columns:
        results['Timestamp'] = pd.to_datetime(results['Timestamp'])
        results = results.sort_values(by='Timestamp')
        
        results['Defective'] = (results['Prediction'] > 0.5).astype(int)
        defect_trends = results.groupby(results['Timestamp'].dt.date)['Defective'].sum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(defect_trends.index, defect_trends.values, marker='o', label='Defective Products')
        plt.title("Defective Products Trend")
        plt.xlabel("Date")
        plt.ylabel("Count of Defective Products")
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(output_dir, "defect_trends.png")
        plt.savefig(output_path)
        print(f"Defect trends saved to {output_path}")
        plt.close()
    else:
        print("Timestamp and/or Prediction columns are missing in the results file.")

def generate_report(output_dir, report_file):
    """
    Combines generated visualizations into a single PDF report.

    Args:
        output_dir (str): Directory containing visualization images.
        report_file (str): Path to save the final PDF report.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add a title page
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Defect Detection Report", ln=True, align='C')
    pdf.ln(10)

    # Add visualizations
    for image_name in ["confusion_matrix.png", "performance_metrics.png", "defect_trends.png"]:
        image_path = os.path.join(output_dir, image_name)
        if os.path.exists(image_path):
            pdf.add_page()
            pdf.image(image_path, x=10, y=20, w=190)
        else:
            print(f"Warning: {image_name} not found, skipping.")

    # Save the PDF
    pdf.output(report_file)
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    # Ensure the output directory exists
    if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
        os.makedirs(VISUALIZATION_OUTPUT_DIR)
    
    # Load results
    print("Loading results...")
    results = load_results(RESULTS_FILE)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_confusion_matrix(results, VISUALIZATION_OUTPUT_DIR)
    plot_performance_metrics(results, VISUALIZATION_OUTPUT_DIR)
    plot_defect_trends(results, VISUALIZATION_OUTPUT_DIR)
    
    # Generate a PDF report
    print("Generating report...")
    generate_report(VISUALIZATION_OUTPUT_DIR, REPORT_FILE)
    print("Visualization and reporting complete.")
