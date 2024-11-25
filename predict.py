from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask('Customer_conversion_rate')

# Load the saved model
model = joblib.load('model/best_pipeline.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Convert input JSON to a numpy array (adjust based on your model's input requirements)
        input_array = np.array(input_data['data'])

        # Make a prediction
        prediction = model.predict(input_array)

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
