import joblib
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# Create a Flask app instance
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# This is a placeholder for the feature names. In a real application,
# you would ideally save these with your model or derive them from your training data.
# For this example, we'll use the feature names from the `X` DataFrame in the notebook.
# If `X` is not available, you would need to manually define the feature names in the correct order.
# Ensure these match the order of features used during model training.
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Create a list of input values in the correct order based on feature_names
        input_values = [data[feature] for feature in feature_names]
        input_data = np.array([input_values])

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': float(prediction_proba[0][0]),
            'probability_diabetes': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # This block allows you to run the app directly using `python app.py`
    # For deployment in Colab or other environments, you might use 'flask run'
    app.run(debug=True, host='0.0.0.0', port=5000)
