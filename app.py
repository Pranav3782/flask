from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
heart_data = pd.read_csv('heart_disease_data.csv')

# Prepare the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [
        int(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        int(request.form['trestbps']),
        int(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        int(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope']),
        int(request.form['ca']),
        int(request.form['thal'])
    ]
    
    # Convert data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    
    # Output result
    if prediction[0] == 0:
        result = 'The Person does not have Heart Disease'
    else:
        result = 'The Person has Heart Disease'
    
    return render_template('index.html', prediction_text=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
