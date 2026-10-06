
# Predictive Analytics for Early Detection of Cardiovascular Diseases Using Machine Learning Techniques

This project predicts the likelihood of cardiovascular disease based on various health parameters. The model uses an ensemble of machine learning algorithms to improve accuracy and provides a user-friendly web interface built with Flask.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Ensembling Technique](#ensembling-technique)
- [User Interface](#user-interface)
- [Results and Evaluation](#results-and-evaluation)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project leverages machine learning algorithms such as Logistic Regression, Random Forest, Support Vector Machine, K-Nearest Neighbours, and XGBoost to predict the presence of cardiovascular diseases. A voting ensemble aggregates predictions from all models to maximize accuracy. The Flask framework is used to deliver a clean, browser-based interface where users can input their health data and view prediction results instantly.

---

## Features

- **Data Preprocessing**: Cleans, encodes, and scales input data.
- **Model Training**: Trains and optimizes five machine learning models.
- **Voting Ensemble**: Combines all models using soft voting for improved performance.
- **Flask Web Interface**: Allows users to input data and get predictions via a browser.
- **Model Evaluation**: Provides classification metrics and ROC visualization.

---

## Setup and Installation

1. **Clone the repository**:
```bash
git clone https://github.com/karthikEdara24/Cardiovascular-Disease-Prediction-Flask.git
cd Cardiovascular-Disease-Prediction-Flask
````

2. **Create and activate virtual environment**:

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Train the model**:

```bash
python app/model/train_model.py
```

---

## Usage

1. **Start the Flask application**:

```bash
python main.py
```

2. **Access the web app**:
   Open your browser and go to: `http://127.0.0.1:5000/`

3. **Input health data** and receive prediction result on cardiovascular risk.

---

## Data Preprocessing

* **Handling Missing Values**: Ensures no null entries in the dataset.
* **Encoding Categorical Variables**: Converts categories like `Sex`, `ChestPainType`, etc., to numeric form.
* **Feature Scaling**: Normalization is applied for better model performance.
* **Feature Selection**: Recursive Feature Elimination (RFE) is used to select the top 8 features.

---

## Models

Five ML models are implemented and tuned:

* **Logistic Regression**
* **Random Forest**
* **Support Vector Machine**
* **K-Nearest Neighbours**
* **XGBoost**

Hyperparameters are optimized using `GridSearchCV` and `RandomizedSearchCV`.

---

## Ensembling Technique

A soft voting classifier combines the models:

* Uses predicted probabilities to make a final decision.
* Enhances model generalization and reduces individual biases.

---

## User Interface

Built using Flask and HTML:

* Users fill in a form with health attributes.
* Output displays whether the user is at **high risk** or **low/no risk**.
* Optionally styled using custom CSS (can be extended).

---

## Results and Evaluation

The ensemble model is evaluated using:

* **Accuracy**: Overall prediction correctness.
* **Precision**: Correct positive predictions.
* **Recall**: Actual positive cases identified.
* **F1-Score**: Balance between precision and recall.
* **ROC & AUC**: Visualization of performance across thresholds.

Sample Metrics:

* Accuracy: `85.87`
* Precision: `86.3`
* Recall: `85.87`
* F1 Score: `85.94`
* AUC Score: Visualized via ROC Curve.

---

## Acknowledgements

* **Dataset**: Publicly available heart disease dataset.
* **Libraries**: Thanks to the teams behind Flask, Pandas, Scikit-learn, XGBoost, and Matplotlib.
* **Contributors**: [Karthik Edara](https://github.com/karthikEdara24), [Mannem Pardhava](https://github.com/Pardhu0007)

> ⚠️ *Note: This project is for educational purposes only and is not intended to be used as a diagnostic tool.*




