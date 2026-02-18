### Employee Attrition Prediction App

A machine learning–based web application built using Streamlit to predict employee attrition. This project helps HR teams identify employees who are at risk of leaving the organization and take preventive actions.

## Project Description

Employee attrition leads to increased hiring costs and loss of valuable talent. This application uses a trained XGBoost model to analyze employee data and predict whether an employee is likely to leave.
The app is designed to be user-friendly, error-safe, and HR-ready, even when some employee data columns are missing.

## Features
Upload employee data (CSV or Excel)
Predict employee attrition (Yes / No)
Shows probability score for each employee
Handles missing or extra columns safely
Displays attrition distribution charts
Feature importance visualization
Download prediction results as CSV

## Tech Stack
Python
Streamlit
Pandas
Scikit-learn
XGBoost
Matplotlib

## Project Structure
employee-attrition/
│
├── models/
│   └── xgboost_model.pkl
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
└── README.md


Clone the repository

git clone https://github.com/simidubey29/employee-attrition.git
cd employee-attrition

## Output
Attrition prediction for each employee
Probability of attrition
Downloadable results file for HR analysis

## Benefits
Helps HR make data-driven decisions
Reduces employee turnover cost

Easy-to-use interface

Works even with partial employee data
