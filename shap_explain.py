import shap
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_stacked_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load feature names
data = pd.read_csv('diabetes.csv')
if 'Pregnancies' in data.columns:
    data.drop('Pregnancies', axis=1, inplace=True)
feature_names = data.drop('Outcome', axis=1).columns

def explain_prediction(input_data):
    # Scale the input
    input_scaled = scaler.transform(pd.DataFrame([input_data], columns=feature_names))
    
    # Get the base RandomForest model
    base_model = model.named_estimators_['rf']
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(base_model)
    
    # Compute SHAP values - get values for class 1 (diabetes positive)
    shap_values = explainer.shap_values(input_scaled)[1][0]
    
    # Calculate feature impacts
    feature_impact = dict(zip(feature_names, shap_values))
    total_impact = np.sum(np.abs(list(feature_impact.values()))) + 1e-8
    
    return {
        'shap_values': shap_values,
        'feature_impact': feature_impact,
        'total_impact': total_impact
    }