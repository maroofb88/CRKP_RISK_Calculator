#!/usr/bin/env python3
"""
CRKP RISK CALCULATOR - Web Application (Complete Features)
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

# Load model
@st.cache_resource
def load_model():
    """Load the corrected model"""
    try:
        model_path = "crkp_model_no_icu.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success(f"✓ Model loaded from {model_path}")
            return model
        else:
            st.error("❌ Model file not found.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Get feature names from model
@st.cache_data
def get_model_features(model):
    """Get list of features expected by the model"""
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        
        # Try to get features from the pipeline
        if hasattr(model, 'named_steps'):
            if 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                # This is complex - we'll use a default list
                pass
        
        # Default feature list (extracted from error message)
        return [
            'age', 'gender', 'age_65_80', 'age_gt_80', 'hemoglobin_min', 'hemoglobin_max', 
            'hemoglobin_mean', 'hemoglobin_last', 'crp_last', 'creatinine_last', 'wbc_last',
            'crp_missing', 'creatinine_missing', 'wbc_missing', 'hemoglobin_missing',
            'has_procedure_record', 'albumin_min', 'carbapenem_30d', 'platelets_max',
            'ast_last', 'dialysis', 'dialysis_30d', 'immunosuppression', 'alt_last',
            'ward_transfers_30d', 'platelets_last', 'diabetes', 'central_line', 'stroke',
            'ast_max', 'lymphocytes_missing', 'creatinine_mean', 'alt_max', 'ventilation',
            'alt_missing', 'neutrophils_min', 'creatinine_max', 'age_50_65', 'age_ge_80',
            'surgery_30d', 'crp_mean', 'wbc_mean', 'ast_mean', 'platelets_min',
            'platelets_missing', 'has_comorbidity_record', 'age_missing', 'any_antibiotic_30d',
            'central_line_30d', 'albumin_last', 'ckd', 'wbc_min', 'neutrophils_mean',
            'creatinine_min', 'has_medication_record', 'albumin_max', 'alt_mean',
            'lymphocytes_min', 'lymphocytes_max', 'gender_missing', 'days_since_last_abx',
            'broad_spectrum_30d', 'crp_max', 'albumin_mean', 'ventilation_30d', 'ward_transfers',
            'cancer', 'platelets_mean', 'neutrophils_last', 'surgery', 'lymphocytes_mean',
            'neutrophils_missing', 'has_lab_record', 'ast_missing', 'age_lt_50', 'cirrhosis',
            'alt_min', 'chf', 'has_transfer_record', 'neutrophils_max', 'hypertension',
            'crp_min', 'lymphocytes_last', 'cad', 'wbc_max', 'ast_min', 'albumin_missing', 'copd'
        ]
    except:
        # Return minimal feature set if we can't determine
        return []

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = []

def create_complete_input_form():
    """Create input form with all required features"""
    st.header("📋 Patient Information")
    
    # Create tabs for different feature categories
    tab1, tab2, tab3 = st.tabs(["Demographics", "Laboratory Values", "Clinical History"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=65, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            gender_numeric = 1 if gender == "Male" else (0 if gender == "Female" else 2)
            
        with col2:
            # Age categories
            age_lt_50 = 1 if age < 50 else 0
            age_50_65 = 1 if 50 <= age < 65 else 0
            age_65_80 = 1 if 65 <= age <= 80 else 0
            age_ge_80 = 1 if age >= 80 else 0
            age_gt_80 = 1 if age > 80 else 0
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Complete Blood Count")
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
            wbc = st.number_input("WBC (×10⁹/L)", min_value=0.1, max_value=50.0, value=8.0, step=0.1)
            platelets = st.number_input("Platelets (×10⁹/L)", min_value=10.0, max_value=1000.0, value=250.0, step=10.0)
            
            # Neutrophils/Lymphocytes (simplified)
            neutrophils = st.number_input("Neutrophils (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            lymphocytes = st.number_input("Lymphocytes (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        
        with col2:
            st.subheader("Chemistry & Inflammation")
            crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
            creatinine = st.number_input("Creatinine (μmol/L)", min_value=10.0, max_value=1000.0, value=80.0, step=1.0)
            albumin = st.number_input("Albumin (g/L)", min_value=10.0, max_value=60.0, value=40.0, step=0.1)
            alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=25.0, step=1.0)
            ast = st.number_input("AST (U/L)", min_value=5.0, max_value=500.0, value=25.0, step=1.0)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Comorbidities")
            hypertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            ckd = st.checkbox("Chronic Kidney Disease")
            cad = st.checkbox("Coronary Artery Disease")
            chf = st.checkbox("Congestive Heart Failure")
            copd = st.checkbox("COPD")
            stroke = st.checkbox("Stroke")
            cirrhosis = st.checkbox("Cirrhosis")
            cancer = st.checkbox("Cancer")
        
        with col2:
            st.subheader("Procedures & Devices (Past 30 days)")
            surgery = st.checkbox("Surgery")
            central_line = st.checkbox("Central Line")
            ventilation = st.checkbox("Mechanical Ventilation")
            dialysis = st.checkbox("Dialysis")
            any_antibiotic = st.checkbox("Any Antibiotic")
            broad_spectrum = st.checkbox("Broad Spectrum Antibiotic")
            carbapenem = st.checkbox("Carbapenem")
            ward_transfers = st.checkbox("Ward Transfers")
            
            # Other records
            has_procedure = st.checkbox("Has Procedure Record")
            has_medication = st.checkbox("Has Medication Record")
            has_lab = st.checkbox("Has Lab Record")
            has_comorbidity = st.checkbox("Has Comorbidity Record")
            has_transfer = st.checkbox("Has Transfer Record")
    
    # Create complete feature dictionary
    features = {
        # Demographics
        'age': float(age),
        'gender': gender_numeric,
        'age_lt_50': age_lt_50,
        'age_50_65': age_50_65,
        'age_65_80': age_65_80,
        'age_ge_80': age_ge_80,
        'age_gt_80': age_gt_80,
        
        # Laboratory values (using same value for min/max/mean for simplicity)
        'hemoglobin_min': float(hemoglobin - 1),
        'hemoglobin_max': float(hemoglobin + 1),
        'hemoglobin_mean': float(hemoglobin),
        'hemoglobin_last': float(hemoglobin),
        
        'wbc_min': float(wbc - 2),
        'wbc_max': float(wbc + 2),
        'wbc_mean': float(wbc),
        'wbc_last': float(wbc),
        
        'platelets_min': float(platelets - 50),
        'platelets_max': float(platelets + 50),
        'platelets_mean': float(platelets),
        'platelets_last': float(platelets),
        
        'neutrophils_min': float(neutrophils - 5),
        'neutrophils_max': float(neutrophils + 5),
        'neutrophils_mean': float(neutrophils),
        'neutrophils_last': float(neutrophils),
        
        'lymphocytes_min': float(lymphocytes - 5),
        'lymphocytes_max': float(lymphocytes + 5),
        'lymphocytes_mean': float(lymphocytes),
        'lymphocytes_last': float(lymphocytes),
        
        'crp_min': float(max(0, crp - 5)),
        'crp_max': float(crp + 5),
        'crp_mean': float(crp),
        'crp_last': float(crp),
        
        'creatinine_min': float(max(0, creatinine - 10)),
        'creatinine_max': float(creatinine + 10),
        'creatinine_mean': float(creatinine),
        'creatinine_last': float(creatinine),
        
        'albumin_min': float(albumin - 2),
        'albumin_max': float(albumin + 2),
        'albumin_mean': float(albumin),
        'albumin_last': float(albumin),
        
        'alt_min': float(max(0, alt - 5)),
        'alt_max': float(alt + 5),
        'alt_mean': float(alt),
        'alt_last': float(alt),
        
        'ast_min': float(max(0, ast - 5)),
        'ast_max': float(ast + 5),
        'ast_mean': float(ast),
        'ast_last': float(ast),
        
        # Missing indicators (assume not missing since we're entering values)
        'age_missing': 0,
        'gender_missing': 0,
        'hemoglobin_missing': 0,
        'wbc_missing': 0,
        'platelets_missing': 0,
        'neutrophils_missing': 0,
        'lymphocytes_missing': 0,
        'crp_missing': 0,
        'creatinine_missing': 0,
        'albumin_missing': 0,
        'alt_missing': 0,
        'ast_missing': 0,
        
        # Comorbidities (1=True, 0=False)
        'hypertension': 1 if hypertension else 0,
        'diabetes': 1 if diabetes else 0,
        'ckd': 1 if ckd else 0,
        'cad': 1 if cad else 0,
        'chf': 1 if chf else 0,
        'copd': 1 if copd else 0,
        'stroke': 1 if stroke else 0,
        'cirrhosis': 1 if cirrhosis else 0,
        'cancer': 1 if cancer else 0,
        
        # Procedures & Devices
        'surgery': 1 if surgery else 0,
        'surgery_30d': 1 if surgery else 0,
        'central_line': 1 if central_line else 0,
        'central_line_30d': 1 if central_line else 0,
        'ventilation': 1 if ventilation else 0,
        'ventilation_30d': 1 if ventilation else 0,
        'dialysis': 1 if dialysis else 0,
        'dialysis_30d': 1 if dialysis else 0,
        'any_antibiotic_30d': 1 if any_antibiotic else 0,
        'broad_spectrum_30d': 1 if broad_spectrum else 0,
        'carbapenem_30d': 1 if carbapenem else 0,
        'ward_transfers': 1 if ward_transfers else 0,
        'ward_transfers_30d': 1 if ward_transfers else 0,
        
        # Other flags
        'has_procedure_record': 1 if has_procedure else 0,
        'has_medication_record': 1 if has_medication else 0,
        'has_lab_record': 1 if has_lab else 0,
        'has_comorbidity_record': 1 if has_comorbidity else 0,
        'has_transfer_record': 1 if has_transfer else 0,
        
        # Set defaults for remaining features
        'immunosuppression': 0,
        'days_since_last_abx': 999,  # Large number = no recent antibiotics
    }
    
    return features

def make_prediction(model, features, expected_features):
    """Make prediction using the model"""
    try:
        # Create DataFrame with all expected features
        features_df = pd.DataFrame(columns=expected_features)
        
        # Add our features
        for feature in expected_features:
            if feature in features:
                features_df[feature] = [features[feature]]
            else:
                # Set default value for missing features
                if feature.endswith('_missing'):
                    features_df[feature] = [1]  # Assume missing
                else:
                    features_df[feature] = [0]  # Default to 0
        
        # Ensure correct data types
        features_df = features_df.astype(float)
        
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
        st.error(f"Number of features provided: {len(features)}")
        st.error(f"Number of features expected: {len(expected_features)}")
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
        if st.session_state.risk_score > 0.01:
            nnt = int(1 / st.session_state.risk_score)
            st.metric(label="Number Needed to Test", value=str(nnt))
        else:
            st.metric(label="Number Needed to Test", value=">100")
    
    # Risk visualization
    st.subheader("Risk Visualization")
    risk_percent = st.session_state.risk_score * 100
    
    # Create a simple gauge
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Risk'], [100], color='lightgray')
    ax.barh(['Risk'], [risk_percent], color=st.session_state.color)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Risk Percentage')
    ax.set_title(f'CRKP Risk: {risk_percent:.1f}%')
    
    # Add threshold lines
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=30, color='orange', linestyle='--', alpha=0.5)
    
    st.pyplot(fig)
    
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
        - Consider empirical antibiotic coverage for resistant organisms
        - Prompt microbiological testing
        """)
    else:  # High risk
        st.error("""
        **Recommended Action:**
        - Contact precautions (gown and gloves)
        - Empirical antibiotic therapy covering CRKP
        - Urgent microbiological testing and susceptibility profiling
        - Infectious disease consultation
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
        st.markdown("""
        ### About
        **Version:** 1.0 (Corrected Model)
        **Model:** XGBoost without ICU features
        
        ### ⚠️ Disclaimer
        **FOR RESEARCH USE ONLY**
        """)
    
    # Main content
    if app_mode == "📋 Input Data":
        st.header("👤 Patient Data Entry")
        
        # Load model
        model = load_model()
        
        if model is not None:
            # Get expected features
            expected_features = get_model_features(model)
            st.session_state.model_features = expected_features
            
            if expected_features:
                st.info(f"Model expects {len(expected_features)} features")
            
            features = create_complete_input_form()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True):
                    with st.spinner("Calculating risk score..."):
                        prediction_prob, risk_category, color = make_prediction(
                            model, features, expected_features
                        )
                        if prediction_prob is not None:
                            st.success(f"✓ Prediction complete!")
                            st.rerun()
            
            with col2:
                if st.button("🔄 Reset Form", type="secondary", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.session_state.risk_score = None
                    st.rerun()
    
    elif app_mode == "📊 View Results":
        if st.session_state.prediction_made:
            display_results()
            
            if st.button("🔄 New Calculation"):
                st.session_state.prediction_made = False
                st.session_state.risk_score = None
                st.rerun()
        else:
            st.warning("No prediction made yet. Please go to 'Input Data' to calculate risk.")
    
    elif app_mode == "ℹ️ Model Info":
        st.header("Model Information")
        
        with st.expander("Performance Metrics"):
            st.markdown("""
            - **AUROC:** 0.613
            - **AUPRC:** 0.263
            - **Brier Score:** 0.222
            """)
        
        with st.expander("Key Predictors"):
            st.markdown("""
            1. **Hemoglobin levels** (min/max/mean)
            2. **Age** categories
            3. **CRP levels**
            4. **Comorbidities and procedures**
            """)
        
        with st.expander("Important Limitations"):
            st.markdown("""
            ⚠️ **ICU data excluded** due to data quality issues
            
            ⚠️ **Other limitations:**
            - Performance limited without ICU data
            - External validation needed
            - For research use only
            """)

# Run the app
if __name__ == "__main__":
    main()
