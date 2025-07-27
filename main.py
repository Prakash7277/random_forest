import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Import joblib for saving/loading models

# --- 1. Prepare the Dataset ---
# For demonstration purposes, we'll create a synthetic dataset.
# In a real-world scenario, you would load your data from a CSV, Excel, or database.

# Sample Data:
# 'Study_Time_Hours': Hours a student studies per day/week
# 'Marks_Percentage': Percentage marks in a recent exam
# 'CGPA': Cumulative Grade Point Average (out of 10 or 4, adjust as needed)
# 'Placement': Target variable (1 for placed, 0 for not placed)

data = {
    'Study_Time_Hours': [2, 3, 5, 4, 6, 1, 3, 7, 5, 2, 4, 6, 8, 3, 5, 1, 4, 7, 6, 2,
                         3, 5, 4, 6, 1, 3, 7, 5, 2, 4, 6, 8, 3, 5, 1, 4, 7, 6, 2, 3],
    'Marks_Percentage': [60, 70, 85, 75, 90, 50, 65, 95, 80, 55, 70, 88, 92, 62, 78, 45, 68, 91, 87, 58,
                         63, 72, 86, 77, 91, 52, 67, 96, 81, 56, 71, 89, 93, 64, 79, 48, 69, 92, 88, 59],
    'CGPA': [6.5, 7.0, 8.5, 7.8, 9.0, 5.5, 6.8, 9.2, 8.2, 6.0, 7.5, 8.9, 9.5, 6.7, 8.0, 5.0, 7.2, 9.1, 8.8, 6.3,
             6.6, 7.1, 8.7, 7.9, 9.1, 5.6, 6.9, 9.3, 8.3, 6.1, 7.6, 9.0, 9.6, 6.8, 8.1, 5.1, 7.3, 9.2, 8.9, 6.4],
    'Placement': [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                  0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0] # 1 for placed, 0 for not placed
}

df = pd.DataFrame(data)

# You would typically load your data like this:
# df = pd.read_csv('your_student_data.csv')
# df = pd.read_excel('your_student_data.xlsx')

print("--- Sample Data Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info()
print("\n--- Data Description ---")
print(df.describe())

# --- 1.5 Exploratory Data Analysis (EDA) ---
print("\n--- Performing Exploratory Data Analysis (EDA) ---")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check unique values for 'Placement' (target variable)
print("\nUnique values in 'Placement' column:")
print(df['Placement'].value_counts())

# Visualize distributions of features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['Study_Time_Hours'], kde=True)
plt.title('Distribution of Study Time (Hours)')

plt.subplot(1, 3, 2)
sns.histplot(df['Marks_Percentage'], kde=True)
plt.title('Distribution of Marks Percentage')

plt.subplot(1, 3, 3)
sns.histplot(df['CGPA'], kde=True)
plt.title('Distribution of CGPA')

plt.tight_layout()
plt.show()

# Visualize relationship between features and target
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='Placement', y='Study_Time_Hours', data=df)
plt.title('Study Time vs. Placement')

plt.subplot(1, 3, 2)
sns.boxplot(x='Placement', y='Marks_Percentage', data=df)
plt.title('Marks Percentage vs. Placement')

plt.subplot(1, 3, 3)
sns.boxplot(x='Placement', y='CGPA', data=df)
plt.title('CGPA vs. Placement')

plt.tight_layout()
plt.show()

# Correlation matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')
plt.show()


# --- 2. Define Features (X) and Target (y) ---
# Features are the input variables used to make predictions.
# Target is the output variable we want to predict.
X = df[['Study_Time_Hours', 'Marks_Percentage', 'CGPA']]
y = df['Placement']

# --- 3. Split the Data into Training and Testing Sets ---
# We split the data to evaluate the model's performance on unseen data.
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 4. Initialize and Train the Random Forest Classifier ---
# RandomForestClassifier is an ensemble learning method for classification.
# n_estimators: The number of trees in the forest. More trees generally lead to better performance but take longer.
# random_state: For reproducibility.
print("\n--- Training the Random Forest Model ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Make Predictions on the Test Set ---
y_pred = model.predict(X_test)

# --- 6. Evaluate the Model ---
# We use various metrics to assess how well our model performed.

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# --- 7. Make a Prediction for a New Student (Example) ---
print("\n--- Predicting for a New Student ---")
# Let's say a new student has:
# Study Time: 4 hours
# Marks: 75%
# CGPA: 7.9

new_student_data = pd.DataFrame([[4, 75, 7.9]], columns=['Study_Time_Hours', 'Marks_Percentage', 'CGPA'])
prediction = model.predict(new_student_data)
prediction_proba = model.predict_proba(new_student_data) # Probability of each class

if prediction[0] == 1:
    print(f"Based on the input, the student is predicted to be PLACED.")
else:
    print(f"Based on the input, the student is predicted to be NOT PLACED.")

print(f"Probability of NOT PLACED (0): {prediction_proba[0][0]:.2f}")
print(f"Probability of PLACED (1): {prediction_proba[0][1]:.2f}")

# --- 8. Feature Importance (Optional but Recommended) ---
# Random Forest models can tell us which features were most important in making predictions.
print("\n--- Feature Importances ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

# --- 9. Save the Trained Model ---
# It's good practice to save the model after training so you don't have to retrain it every time.
# The model will be saved as 'random_forest_model.pkl' in the same directory.
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")


