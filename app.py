import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🩺 AI-Powered Diabetes Assistant",
    page_icon="🩺",
    layout="centered"
)

# Load assets
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets['model'] = joblib.load("diabetes_stacked_model.pkl")
        assets['scaler'] = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()
    return assets

# Clinical form layout
def create_clinical_form():
    with st.form("clinical_form"):
        st.header("Patient Clinical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numerical inputs with validation
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=50,
                max_value=300,
                value=120,
                step=1,
                help="Normal range: 70-99 mg/dL"
            )
            
            bp_options = ["<80 (Normal)", "80-89 (Elevated)", "≥90 (Hypertension)"]
            blood_pressure = st.selectbox(
                "Blood Pressure Category",
                bp_options,
                index=0,
                help="Diastolic pressure classification"
            )
            
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                format="%.1f"
            )
            
        with col2:
            insulin = st.number_input(
                "Insulin (μIU/mL)",
                min_value=0,
                max_value=900,
                value=80,
                step=1
            )
            
            age_group = st.radio(
                "Age Group",
                options=["<30", "30-45", "45-65", "≥65"],
                horizontal=True
            )
            
            skin_thickness = st.select_slider(
                "Skin Thickness (mm)",
                options=[7, 10, 15, 20, 25, 30, 35, 40, 50, 100],
                value=20
            )
        
        # Convert categorical inputs to numerical values
        bp_map = {"<80 (Normal)": 75, "80-89 (Elevated)": 85, "≥90 (Hypertension)": 95}
        age_map = {"<30": 25, "30-45": 38, "45-65": 55, "≥65": 70}
        
        input_values = {
            'Glucose': glucose,
            'BloodPressure': bp_map[blood_pressure],
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': 0.5,  # Default value
            'Age': age_map[age_group]
        }
        
        submitted = st.form_submit_button("Analyze Diabetes Risk", type="primary")
        
        return submitted, input_values

def show_results(prediction, probability):
    if prediction == 1:
        st.error(f"""
        ## 🚨 High Diabetes Risk
        **Confidence:** {probability*100:.1f}%  
        **Recommendation:** Consult a healthcare provider for confirmatory tests
        """)
    else:
        st.success(f"""
        ## ✅ Low Diabetes Risk
        **Confidence:** {probability*100:.1f}%  
        **Recommendation:** Maintain healthy lifestyle with regular checkups
        """)

def main():
    assets = load_assets()
    
    st.title("🩺 Clinical Diabetes Risk Assessment")
    st.markdown("---")
    
    submitted, input_values = create_clinical_form()
    
    if submitted:
        with st.spinner("Analyzing clinical parameters..."):
            try:
                # Prepare input in correct feature order
                feature_order = ['Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                input_array = np.array([[input_values[feat] for feat in feature_order]])
                
                # Scale and predict
                input_scaled = assets['scaler'].transform(input_array)
                prediction = assets['model'].predict(input_scaled)[0]
                proba = assets['model'].predict_proba(input_scaled)[0][prediction]
                
                st.markdown("---")
                show_results(prediction, proba)
                
                # Show input summary
                with st.expander("See entered values"):
                    st.table(pd.DataFrame.from_dict(input_values, orient='index', columns=['Value']))
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()