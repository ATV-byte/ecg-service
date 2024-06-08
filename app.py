from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import csv
from scipy import stats
from tensorflow.keras.models import load_model
import time
import io

app = Flask(__name__)

# Denoise function (dummy implementation for placeholder)
def denoise(signal):
    # Implement your denoising function here
    return signal

@app.route('/get-likelihood', methods=['POST'])
def get_likelihood():
    start_time = time.time()
    
    # Load the model
    model = load_model('ecg_classification_model.keras')
    
    # Read the CSV file from the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if file:
        # Read the file content
        file_content = file.read().decode('utf-8')
        signals = []
        csvfile = io.StringIO(file_content)
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row_index, row in enumerate(spamreader):
            if row_index > 0:
                signals.append(int(row[1]))  # Assuming ECG data is in the second column
        
        # Preprocess the signals
        signals = denoise(signals)
        signals = stats.zscore(signals)
        
        # Split the signals into windows
        window_size = 180
        X = []
        for pos in range(window_size, len(signals) - window_size):
            beat = signals[pos - window_size:pos + window_size]
            X.append(beat)
        
        # Convert to numpy array and reshape for the model
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Batch processing for predictions
        batch_size = 2048
        num_batches = X.shape[0] // batch_size + 1  # Calculate total number of batches
        predictions = []
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i + batch_size]
            batch_predictions = model.predict(batch)
            predictions.append(batch_predictions)
        
        predictions = np.vstack(predictions)
        
        # Calculate the likelihood of heart issues (sum of non-normal classes)
        heart_issue_prob = np.sum(predictions[:, 1:], axis=1)
        avg_heart_issue_prob = np.mean(heart_issue_prob)
        
        end_time = time.time()
        
        response = {
            "likelihood_of_heart_issues": avg_heart_issue_prob * 100,
            "time_taken_for_scoring": end_time - start_time
        }
        
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
