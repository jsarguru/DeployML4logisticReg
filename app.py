from flask import jsonify
import numpy as np

app = Flask(__name__)
print("Flask application structure initialized.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming the input data is a dictionary where keys are feature names
        # and values are the corresponding data.
        # It should match the order of features used during training.
        
        # Retrieve feature names from the original DataFrame X
        feature_names = X.columns.tolist()

        # Create a list of input values in the correct order
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

print("Prediction endpoint defined.")
