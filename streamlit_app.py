#!/usr/bin/env python3
"""
CRKP RISK CALCULATOR - Web Application
Deploy corrected model (no ICU features) as web calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CRKP Risk Calculator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🦠 CRKP Risk Calculator")
st.markdown("""
### Clinical Decision Support Tool
This calculator estimates the risk of Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP) infection using machine learning.

**⚠️ Important Disclaimer:** This tool is for research use only. Clinical decisions should be made by qualified healthcare professionals.
""")

# Load model - FIXED PATH for Streamlit Cloud
@st.cache_resource
def load_model():
    """Load the corrected model"""
    try:
        # Try multiple possible paths
        possible_paths = [
            "crkp_model_no_icu.pkl",  # Same directory
            "./crkp_model_no_icu.pkl",  # Current directory
            "model_no_icu/crkp_model_no_icu.pkl"  # Subdirectory
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success(f"✓ Model loaded from {model_path}")
                return model
        
        # If we get here, model not found
        st.error("❌ Model file not found. Available files:")
        for f in os.listdir('.'):
            st.write(f"  - {f}")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None

def create_input_form():
    """Create input form for clinical features"""
    st.header("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=65, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        
        # Convert gender to numeric
        gender_numeric = 1 if gender == "Male" else (0 if gender == "Female" else 2)
        
        # Age categories
        age_65_80 = 1 if 65 <= age <= 80 else 0
        age_gt_80 = 1 if age > 80 else 0
    
    with col2:
        st.subheader("Laboratory Values")
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
        crp = st.number_input("C-reactive Protein (mg/L)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
        creatinine = st.number_input("Creatinine (μmol/L)", min_value=10.0, max_value=1000.0, value=80.0, step=1.0)
        wbc = st.number_input("White Blood Cells (×10⁹/L)", min_value=0.1, max_value=50.0, value=8.0, step=0.1)
    
    # Create feature dictionary
    features = {
        'age': float(age),
        'gender': gender_numeric,
        'age_65_80': age_65_80,
        'age_gt_80': age_gt_80,
        'hemoglobin_min': float(hemoglobin - 2),
        'hemoglobin_max': float(hemoglobin + 2),
        'hemoglobin_mean': float(hemoglobin),
        'hemoglobin_last': float(hemoglobin),
        'crp_last': float(crp),
        'creatinine_last': float(creatinine),
        'wbc_last': float(wbc),
        'crp_missing': 0,
        'creatinine_missing': 0,
        'wbc_missing': 0,
        'hemoglobin_missing': 0
    }
    
    # Add procedure history
    has_procedure = st.checkbox("Has procedure record in past 30 days?")
    features['has_procedure_record'] = 1 if has_procedure else 0
    
    return features

def make_prediction(model, features):
    """Make prediction using the model"""
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction_prob = model.predict_proba(features_df)[0, 1]
        
        # Determine risk category
        if prediction_prob < 0.1:
            risk_category = "Low"
            color = "green"
        elif prediction_prob < 0.3:
            risk_category = "Moderate"
            color = "orange"
        else:
            risk_category = "High"
            color = "red"
        
        # Store results
        st.session_state.prediction_made = True
        st.session_state.risk_score = prediction_prob
        st.session_state.risk_category = risk_category
        st.session_state.color = color
        
        return prediction_prob, risk_category, color
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Features used: {features}")
        return None, None, None

def display_results():
    """Display prediction results"""
    st.header("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="CRKP Risk Score",
            value=f"{st.session_state.risk_score:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Risk Category",
            value=st.session_state.risk_category,
            delta=None
        )
    
    with col3:
        # Calculate number needed to treat
        if st.session_state.risk_score > 0:
            nnt = int(1 / st.session_state.risk_score) if st.session_state.risk_score > 0.01 else ">100"
            st.metric(
                label="Number Needed to Test",
                value=str(nnt)
            )
    
    # Risk visualization
    st.subheader("Risk Visualization")
    
    # Simple progress bar
    risk_percent = st.session_state.risk_score * 100
    st.progress(min(risk_percent / 100, 1.0))
    st.caption(f"Risk level: {risk_percent:.1f}%")
    
    # Clinical recommendations
    st.subheader("🎯 Clinical Recommendations")
    
    if st.session_state.risk_category == "Low":
        st.success("""
        **Recommended Action:**
        - Standard infection control precautions
        - Routine microbiological testing if clinically indicated
        """)
    elif st.session_state.risk_category == "Moderate":
        st.warning("""
        **Recommended Action:**
        - Enhanced infection control precautions
        - Consider empirical antibiotic coverage
        - Prompt microbiological testing
        """)
    else:  # High risk
        st.error("""
        **Recommended Action:**
        - Contact precautions (gown and gloves)
        - Empirical antibiotic therapy covering CRKP
        - Urgent microbiological testing
        - Infectious disease consultation
        """)

def display_model_info():
    """Display model information and limitations"""
    with st.expander("🔍 Model Information & Limitations", expanded=False):
        st.markdown("""
        ### Model Details
        - **Algorithm:** XGBoost
        - **Training Data:** 5,780 patients (2012-2023)
        - **Test Data:** 1,445 patients (2023-2024)
        - **Temporal Validation:** 80/20 chronological split
        
        ### Performance Metrics
        - **AUROC:** 0.613 (Area Under ROC Curve)
        - **AUPRC:** 0.263 (Area Under Precision-Recall Curve)
        - **Brier Score:** 0.222
        
        ### Key Predictors
        1. **Hemoglobin levels** (48.7% of importance)
        2. **Age** (16.7% of importance)
        3. **CRP levels** (8.2% of importance)
        
        ### Important Limitations
        ⚠️ **CRITICAL LIMITATION:** ICU admission data was excluded due to data quality issues
        
        ⚠️ **Other Limitations:**
        - Limited to demographic and laboratory predictors
        - External validation not yet performed
        - For research use only
        
        ### Intended Use
        This tool is intended for:
        - Research and quality improvement
        - Clinical decision support (adjunct to clinical judgment)
        - Antibiotic stewardship programs
        """)

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        app_mode = st.radio(
            "Select Mode:",
            ["📋 Input Data", "📊 View Results", "ℹ️ Model Info"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Version:** 1.0 (Corrected Model)
        **Last Updated:** December 2024
        **Model:** XGBoost without ICU features
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        **FOR RESEARCH USE ONLY**
        
        This tool does not provide medical advice.
        Always consult qualified healthcare professionals.
        """)
    
    # Main content
    if app_mode == "📋 Input Data":
        st.header("👤 Patient Data Entry")
        
        # Load model first
        model = load_model()
        
        if model is not None:
            features = create_input_form()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True):
                    with st.spinner("Calculating risk score..."):
                        prediction_prob, risk_category, color = make_prediction(model, features)
                        if prediction_prob is not None:
                            st.success(f"Prediction complete! Risk: {prediction_prob:.1%}")
                            st.rerun()
            
            with col2:
                if st.button("🔄 Reset Form", type="secondary", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.session_state.risk_score = None
                    st.rerun()
    
    elif app_mode == "📊 View Results":
        if st.session_state.prediction_made:
            display_results()
            
            # Export options
            st.subheader("📥 Export Results")
            if st.button("📋 Copy Results to Clipboard"):
                st.code(f"""
                CRKP Risk Assessment Report
                ===========================
                Risk Score: {st.session_state.risk_score:.1%}
                Risk Category: {st.session_state.risk_category}
                Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                
                Disclaimer: For research use only.
                """)
                st.success("Results ready to copy!")
            
            if st.button("🔄 New Calculation"):
                st.session_state.prediction_made = False
                st.session_state.risk_score = None
                st.rerun()
        else:
            st.warning("No prediction made yet. Please go to 'Input Data' to calculate risk.")
    
    elif app_mode == "ℹ️ Model Info":
        display_model_info()

# Run the app
if __name__ == "__main__":
    main()
