from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask('Customer_conversion_rate')

# Load the saved model
model = joblib.load('model/best_pipeline.pkl')

# Selected features
selected_features = ['PagesViewed', 'Age', 'EmailSent', 'TimeSpentMinutes', 'FollowUpEmails', 'SocialMediaEngagement', 'FormSubmissions', 'Downloads', 'ResponseTimeHours','Location', 'LeadStatus',]

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Create dataframe based on JSON object
        df = pd.DataFrame(input_data)       

        # Get only excepted columns from df
        df = df[selected_features]

        # Make a prediction
        prediction = model.predict_proba(df)[0,1]

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define a health check endpoint
@app.route('/home', methods=['GET'])
def home():
    return jsonify({'status': 'ok'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
