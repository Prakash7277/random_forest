# app.py
from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define paths to the model and feature list files
MODEL_PATH = 'random_forest_model.pkl'
FEATURES_PATH = 'model_features.pkl'

model = None # Initialize model as None
model_features = None # Initialize feature list as None

# Load the model and feature list when the Flask application starts
@app.before_request
def load_assets():
    """
    Loads the pre-trained Random Forest model and the list of features.
    This function is called once when the Flask app starts.
    """
    global model, model_features
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                print(f"Model '{MODEL_PATH}' loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
        else:
            print(f"Model file '{MODEL_PATH}' not found. Please run the training script first.")

    if model_features is None:
        if os.path.exists(FEATURES_PATH):
            try:
                model_features = joblib.load(FEATURES_PATH)
                print(f"Model features '{FEATURES_PATH}' loaded successfully.")
            except Exception as e:
                print(f"Error loading model features: {e}")
                model_features = None
        else:
            print(f"Model features file '{FEATURES_PATH}' not found. Please run the training script first.")


# HTML template for the Flask app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Placement Predictor</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6; /* Light gray background */
        }
        /* Custom styles for better aesthetics */
        .card {
            background-color: #ffffff;
            border-radius: 1rem; /* Rounded corners */
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .btn {
            background-color: #4f46e5; /* Indigo 600 */
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem; /* Rounded corners */
            transition: background-color 0.3s ease;
        }
        .btn:hover {
            background-color: #4338ca; /* Indigo 700 */
        }
        input[type="number"], select {
            border: 1px solid #d1d5db; /* Gray 300 */
            border-radius: 0.5rem; /* Rounded corners */
            padding: 0.5rem 1rem;
            width: 100%;
            box-sizing: border-box; /* Include padding in width */
        }
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #6366f1; /* Indigo 500 */
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4">
    <div class="card w-full max-w-md p-8 space-y-6">
        <h1 class="text-3xl font-bold text-center text-gray-800">Student Placement Predictor</h1>

        <form action="/predict" method="post" class="space-y-4">
            <div>
                <label for="study_time" class="block text-sm font-medium text-gray-700 mb-1">Study Time (Hours):</label>
                <input type="number" id="study_time" name="study_time" step="0.1" min="0" required
                       class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm">
            </div>
            <div>
                <label for="marks" class="block text-sm font-medium text-gray-700 mb-1">Marks Percentage (%):</label>
                <input type="number" id="marks" name="marks" step="0.1" min="0" max="100" required
                       class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm">
            </div>
            <div>
                <label for="cgpa" class="block text-sm font-medium text-gray-700 mb-1">CGPA (out of 10):</label>
                <input type="number" id="cgpa" name="cgpa" step="0.01" min="0" max="10" required
                       class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm">
            </div>
            <div>
                <label for="gender" class="block text-sm font-medium text-gray-700 mb-1">Gender:</label>
                <select id="gender" name="gender" required
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <button type="submit" class="btn w-full">Predict Placement</button>
        </form>

        {% if prediction_text %}
            <div class="mt-6 p-4 rounded-lg text-center
                        {% if 'PLACED' in prediction_text %} bg-green-100 text-green-800 {% else %} bg-red-100 text-red-800 {% endif %}">
                <p class="font-semibold text-lg">{{ prediction_text }}</p>
                {% if probability_text %}
                    <p class="text-sm mt-1">{{ probability_text }}</p>
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """
    Renders the home page with the prediction form.
    """
    return render_template_string(HTML_TEMPLATE, prediction_text="", probability_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web form.
    Loads input data, makes a prediction using the loaded model, and returns the result.
    """
    if model is None or model_features is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: Model or feature list not loaded. Please ensure 'random_forest_model.pkl' and 'model_features.pkl' exist.")

    try:
        # Get input data from the form
        study_time = float(request.form['study_time'])
        marks = float(request.form['marks'])
        cgpa = float(request.form['cgpa'])
        gender = request.form['gender'] # 'Male' or 'Female'

        # Create a dictionary to hold the input data, initialized with numerical features
        input_dict = {
            'Study_Time_Hours': study_time,
            'Marks_Percentage': marks,
            'CGPA': cgpa
        }

        # Initialize gender columns to 0
        for feature in model_features:
            if 'Gender_' in feature:
                input_dict[feature] = 0

        # Set the appropriate gender column to 1 based on user input
        if gender == 'Male':
            input_dict['Gender_Male'] = 1
        elif gender == 'Female':
            input_dict['Gender_Female'] = 1
        # Add more elif for other genders if your dataset has them

        # Create a DataFrame from the input dictionary, ensuring column order matches training
        # We create a DataFrame with a single row and then reindex it to match the training features.
        input_df = pd.DataFrame([input_dict])
        input_data = input_df[model_features] # Ensure correct column order

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        result_text = "The student is predicted to be PLACED." if prediction == 1 else "The student is predicted to be NOT PLACED."
        probability_info = f"Probability of PLACED: {prediction_proba[1]:.2f}, NOT PLACED: {prediction_proba[0]:.2f}"

        return render_template_string(HTML_TEMPLATE, prediction_text=result_text, probability_text=probability_info)

    except ValueError:
        return render_template_string(HTML_TEMPLATE, prediction_text="Invalid input. Please enter numerical values for study time, marks, and CGPA.")
    except KeyError as ke:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Missing expected input field: {ke}. Please ensure all fields are provided.")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
