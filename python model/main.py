import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Import joblib for saving/loading models
import os

# --- 1. Load the Dataset ---
# Assuming 'student_placement_data_with_gender.csv' is in the same directory
csv_file_path = 'student_placement_data_with_gender.csv'

if not os.path.exists(csv_file_path):
    print(f"Error: Dataset '{csv_file_path}' not found. Please ensure it's in the same directory.")
    # You might want to exit or handle this more gracefully in a real application
    exit()

df = pd.read_csv(csv_file_path)

print("--- Dataset Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info()
print("\n--- Data Description ---")
print(df.describe(include='all')) # Include 'all' to describe categorical columns too

# --- 1.5 Exploratory Data Analysis (EDA) ---
print("\n--- Performing Exploratory Data Analysis (EDA) ---")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check unique values for 'Placement' (target variable)
print("\nUnique values in 'Placement' column:")
print(df['Placement'].value_counts())

# Check unique values for 'Gender'
print("\nUnique values in 'Gender' column:")
print(df['Gender'].value_counts())

# Visualize distributions of numerical features
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
sns.histplot(df['Study_Time_Hours'], kde=True)
plt.title('Distribution of Study Time (Hours)')

plt.subplot(1, 4, 2)
sns.histplot(df['Marks_Percentage'], kde=True)
plt.title('Distribution of Marks Percentage')

plt.subplot(1, 4, 3)
sns.histplot(df['CGPA'], kde=True)
plt.title('Distribution of CGPA')

plt.subplot(1, 4, 4)
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Gender')

plt.tight_layout()
plt.show()

# Visualize relationship between features and target
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
sns.boxplot(x='Placement', y='Study_Time_Hours', data=df)
plt.title('Study Time vs. Placement')

plt.subplot(1, 4, 2)
sns.boxplot(x='Placement', y='Marks_Percentage', data=df)
plt.title('Marks Percentage vs. Placement')

plt.subplot(1, 4, 3)
sns.boxplot(x='Placement', y='CGPA', data=df)
plt.title('CGPA vs. Placement')

plt.subplot(1, 4, 4)
sns.countplot(x='Gender', hue='Placement', data=df)
plt.title('Placement by Gender')

plt.tight_layout()
plt.show()

# Correlation matrix for numerical features
print("\n--- Correlation Matrix (Numerical Features) ---")
print(df[['Study_Time_Hours', 'Marks_Percentage', 'CGPA', 'Placement']].corr())
sns.heatmap(df[['Study_Time_Hours', 'Marks_Percentage', 'CGPA', 'Placement']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Target')
plt.show()

# --- 2. Feature Engineering: Handle Categorical 'Gender' Column ---
# Convert 'Gender' into numerical format using one-hot encoding.
# This creates new columns like 'Gender_Female' and 'Gender_Male'.
# drop_first=True avoids multicollinearity by dropping one of the gender columns
# (e.g., if Gender_Female is 0, it implies Gender_Male is 1, assuming only two categories).
# However, for simpler interpretation and to ensure all gender categories are explicitly handled
# in the Flask app, we will keep both columns (drop_first=False).
df_encoded = pd.get_dummies(df, columns=['Gender'], prefix='Gender', drop_first=False)

print("\n--- DataFrame after One-Hot Encoding Gender ---")
print(df_encoded.head())
print("\n--- Data Info after Encoding ---")
df_encoded.info()

# --- 3. Define Features (X) and Target (y) ---
# Features now include the encoded 'Gender' columns.
feature_columns = ['Study_Time_Hours', 'Marks_Percentage', 'CGPA'] + [col for col in df_encoded.columns if 'Gender_' in col]
X = df_encoded[feature_columns]
y = df_encoded['Placement']

# Store the list of feature columns for use in the Flask app
# This ensures the Flask app uses the exact same order and names of features.
model_features = X.columns.tolist()
joblib.dump(model_features, 'model_features.pkl') # Save feature list

print(f"\nFeatures used for training: {model_features}")

# --- 4. Split the Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 5. Initialize and Train the Random Forest Classifier ---
print("\n--- Training the Random Forest Model ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Make Predictions on the Test Set ---
y_pred = model.predict(X_test)

# --- 7. Evaluate the Model ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# --- 8. Feature Importance (Optional but Recommended) ---
print("\n--- Feature Importances ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

# --- 9. Save the Trained Model ---
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")

