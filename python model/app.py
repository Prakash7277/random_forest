# app.py
from flask import Flask, request, render_template
import joblib
import pandas as pd
import os


app = Flask(__name__)

# Define the path to the model file. It's assumed to be in the same directory as app.py.
MODEL_PATH = 'random_forest_model.pkl'
model = None # Initialize model as None

# Load the model when the Flask application starts
@app.before_request
def load_model():
    """
    Loads the pre-trained Random Forest model.
    This function is called once when the Flask app starts.
    This function is reposible for loading the pkl file
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                print(f"Model '{MODEL_PATH}' loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None # Ensure model is None if loading fails
        else:
            print(f"Model file '{MODEL_PATH}' not found. Please run the training script first.")

# HTML template for the Flask app
# This includes Tailwind CSS for a responsive and modern look.


@app.route('/')
def home():
    """
    Renders the home page with the prediction form.
    """
    return render_template("index.html", prediction_text="", probability_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web form.
    Loads input data, makes a prediction using the loaded model, and returns the result.
    """
    if model is None:
        return render_template("index.html", prediction_text="Error: Model not loaded. Please ensure 'random_forest_model.pkl' exists.")

    try:
        # Get input data from the form
        study_time = float(request.form['study_time'])
        marks = float(request.form['marks'])
        cgpa = float(request.form['cgpa'])

        # Create a DataFrame for the prediction
        input_data = pd.DataFrame([[study_time, marks, cgpa]],
                                  columns=['Study_Time_Hours', 'Marks_Percentage', 'CGPA'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        result_text = "The student is predicted to be PLACED." if prediction == 1 else "The student is predicted to be NOT PLACED."
        probability_info = f"Probability of PLACED: {prediction_proba[1]:.2f}, NOT PLACED: {prediction_proba[0]:.2f}"

        return render_template("index.html", prediction_text=result_text, probability_text=probability_info)

    except ValueError:
        return render_template("index.html", prediction_text="Invalid input. Please enter numerical values.")
    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {e}")

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, you would use a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True) # debug=True allows for automatic reloading on code changes and shows detailed errors
