import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import sys
import traceback

# Page config
st.set_page_config(
    page_title="CRKP Risk Calculator",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 CRKP Risk Prediction Calculator")
st.markdown("Clinical Decision Support Tool")

# Try to load model with detailed error handling
@st.cache_resource
def load_model():
    try:
        # Try absolute path first
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgb_enn_model.pkl')
        
        st.write(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            # List files in models directory
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            if os.path.exists(models_dir):
                st.write("Files in models directory:", os.listdir(models_dir))
            return None, None, None
        
        model = joblib.load(model_path)
        st.success("✓ Model loaded successfully")
        
        # Try to load preprocessor
        preprocessor_path = os.path.join(os.path.dirname(__file__), 'models', 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            st.warning("Preprocessor not found, using default")
            from sklearn.preprocessing import StandardScaler
            preprocessor = StandardScaler()
        
        # Try to load feature info
        feature_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_info.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_info = json.load(f)
        else:
            feature_info = {'feature_names': ['age', 'gender']}
            
        return model, preprocessor, feature_info
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None

# Load model
model, preprocessor, feature_info = load_model()

# Show model status
if model is None:
    st.warning("⚠️ Using demo mode - enter data below")

# Simple input form
with st.sidebar:
    st.header("Patient Data")
    
    age = st.number_input("Age", 0, 120, 65)
    icu = st.checkbox("Recent ICU admission")
    abx = st.checkbox("Recent antibiotics")
    calculate = st.button("Calculate Risk", type="primary")

if calculate:
    try:
        if model is not None:
            # Real prediction
            # Prepare input data
            input_data = {
                'age': age,
                'gender': 1,  # default male
                'icu_admission_30d': 1 if icu else 0,
                'any_antibiotic_30d': 1 if abx else 0,
                'days_since_last_abx': 7,
                'num_abx_classes_30d': 1,
                'total_abx_days_30d': 7,
                'ward_transfers_30d': 0,
                'days_in_hospital_before_culture': 7,
                'current_location_icu': 1 if icu else 0,
                'lymphocytes_last': 1.5,
                'albumin_last': 3.5,
                'plt_last': 200,
                'age_lt_50': 1 if age < 50 else 0,
                'age_50_65': 1 if 50 <= age < 65 else 0,
                'age_65_80': 1 if 65 <= age < 80 else 0,
                'age_ge_80': 1 if age >= 80 else 0,
                'age_missing': 0,
                'gender_missing': 0,
                'carbapenem_30d': 0,
                'broad_spectrum_30d': 0,
                'cephalosporin_30d': 0,
                'fluoroquinolone_30d': 0
            }
            
            # Create DataFrame
            feature_order = feature_info.get('feature_names', ['age', 'gender'])
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features present
            for feat in feature_order:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            input_df = input_df[feature_order]
            
            # Predict
            X_processed = preprocessor.transform(input_df)
            probability = model.predict_proba(X_processed)[0, 1]
            
        else:
            # Demo mode calculation
            probability = 0.1
            if icu: probability += 0.3
            if abx: probability += 0.2
            if age > 70: probability += 0.1
            probability = min(probability, 0.9)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CRKP Probability", f"{probability:.1%}")
        
        with col2:
            if probability < 0.2:
                risk = "Low"
            elif probability < 0.5:
                risk = "Moderate"
            else:
                risk = "High"
            st.metric("Risk Level", risk)
        
        # Recommendations
        if probability >= 0.263:
            st.warning("Consider carbapenem-sparing therapy and contact precautions")
        else:
            st.success("CRKP unlikely - standard empiric therapy may be appropriate")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("For research use only | Model: XGBoost | AUPRC: 0.269")
