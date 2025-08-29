from flask import render_template, request
from app import app
import pickle
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
rfe = model_data['rfe']
scaler = model_data['scaler']

def transform_input(data):
    arr = np.array([data])
    arr_scaled = scaler.transform(arr)
    arr_selected = rfe.transform(arr_scaled)
    return arr_selected

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = [
            float(request.form['Age']),
            float(request.form['RestingBP']),
            float(request.form['Cholesterol']),
            float(request.form['FastingBS']),
            float(request.form['MaxHR']),
            float(request.form['Oldpeak']),
            float(request.form['Sex']),
            float(request.form['ChestPainType']),
            float(request.form['RestingECG']),
            float(request.form['ExerciseAngina']),
            float(request.form['ST_Slope'])
        ]
        prediction = model.predict(transform_input(form_data))[0]
        message = "High Risk of Cardiovascular Disease" if prediction == 1 else "Low Risk / No Cardiovascular Disease"
        return render_template('result.html', prediction=message)