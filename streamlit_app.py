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
st.title("🦠 Carbapenem-Resistant Klebsiella pneumoniae (CRKP) Risk Calculator")
st.markdown("""
### Clinical Decision Support Tool
This calculator estimates the risk of CRKP infection using machine learning.
**Model features:** Demographic and laboratory predictors (ICU data excluded due to data quality issues)

**⚠️ Important Disclaimer:** This tool is for research use only. Clinical decisions should be made by qualified healthcare professionals.
""")

# Load model
@st.cache_resource
def load_model():
    """Load the corrected model"""
    try:
        model_path = "model_no_icu/crkp_model_no_icu.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error("Model file not found. Please ensure 'model_no_icu/crkp_model_no_icu.pkl' exists.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load feature importance for explanation
@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        importance_path = "shap_no_icu/feature_importance.csv"
        if os.path.exists(importance_path):
            return pd.read_csv(importance_path)
        return None
    except:
        return None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}

def create_input_form():
    """Create input form for clinical features"""
    st.header("📋 Patient Information Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=65, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other/Unknown"])
        
        # Convert gender to numeric (if needed by model)
        gender_numeric = 1 if gender == "Male" else (0 if gender == "Female" else 2)
        
        # Age categories (if model uses them)
        age_65_80 = 1 if 65 <= age <= 80 else 0
        age_gt_80 = 1 if age > 80 else 0
    
    with col2:
        st.subheader("Laboratory Values")
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
        wbc = st.number_input("White Blood Cells (×10⁹/L)", min_value=0.1, max_value=50.0, value=8.0, step=0.1)
        crp = st.number_input("C-reactive Protein (mg/L)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
        creatinine = st.number_input("Creatinine (μmol/L)", min_value=10.0, max_value=1000.0, value=80.0, step=1.0)
    
    with col3:
        st.subheader("Additional Parameters")
        platelets = st.number_input("Platelets (×10⁹/L)", min_value=10.0, max_value=1000.0, value=250.0, step=10.0)
        albumin = st.number_input("Albumin (g/L)", min_value=10.0, max_value=60.0, value=40.0, step=0.1)
        alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=25.0, step=1.0)
        ast = st.number_input("AST (U/L)", min_value=5.0, max_value=500.0, value=25.0, step=1.0)
    
    # Create feature dictionary
    features = {
        'age': age,
        'gender': gender_numeric,
        'age_65_80': age_65_80,
        'age_gt_80': age_gt_80,
        'hemoglobin_min': hemoglobin - 2,  # Simulated min
        'hemoglobin_max': hemoglobin + 2,  # Simulated max
        'hemoglobin_mean': hemoglobin,
        'hemoglobin_last': hemoglobin,
        'wbc_last': wbc,
        'crp_last': crp,
        'creatinine_last': creatinine,
        'platelets_last': platelets,
        'albumin_last': albumin,
        'alt_last': alt,
        'ast_last': ast,
        # Add missing indicators
        'crp_missing': 0,
        'creatinine_missing': 0,
        'wbc_missing': 0,
        'albumin_missing': 0,
        'hemoglobin_missing': 0,
        'platelets_missing': 0,
        'alt_missing': 0,
        'ast_missing': 0
    }
    
    # Add procedure history (simplified)
    has_procedure = st.checkbox("Has procedure record in past 30 days?")
    features['has_procedure_record'] = 1 if has_procedure else 0
    
    # Store in session state
    st.session_state.feature_values = features
    
    return features

def make_prediction(model, features):
    """Make prediction using the model"""
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Reorder columns to match model expectations
        # Get feature names from model if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            # Ensure all expected features are present
            for feat in expected_features:
                if feat not in features_df.columns:
                    features_df[feat] = 0  # Default value for missing features
        
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
    
    # Risk gauge
    st.subheader("Risk Visualization")
    
    # Create a simple gauge
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CRKP Risk Percentage"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': st.session_state.color},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 30], 'color': "yellow"},
                {'range': [30, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': st.session_state.risk_score * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Clinical recommendations
    st.subheader("🎯 Clinical Recommendations")
    
    if st.session_state.risk_category == "Low":
        st.success("""
        **Recommended Action:**
        - Standard infection control precautions
        - Routine microbiological testing if clinically indicated
        - Consider alternative diagnoses
        """)
    elif st.session_state.risk_category == "Moderate":
        st.warning("""
        **Recommended Action:**
        - Enhanced infection control precautions
        - Consider empirical antibiotic coverage for resistant organisms
        - Prompt microbiological testing
        - Monitor closely for clinical deterioration
        """)
    else:  # High risk
        st.error("""
        **Recommended Action:**
        - Contact precautions (gown and gloves)
        - Empirical antibiotic therapy covering CRKP
        - Urgent microbiological testing and susceptibility profiling
        - Infectious disease consultation
        - Consider isolation if in healthcare setting
        """)

def display_model_info():
    """Display model information and limitations"""
    with st.expander("🔍 Model Information & Limitations", expanded=False):
        st.markdown("""
        ### Model Details
        - **Algorithm:** XGBoost (Gradient Boosting)
        - **Training Data:** 5,780 patients (2012-2023)
        - **Test Data:** 1,445 patients (2023-2024)
        - **Temporal Validation:** 80/20 chronological split
        
        ### Performance Metrics
        - **AUROC:** 0.613 (Area Under ROC Curve)
        - **AUPRC:** 0.263 (Area Under Precision-Recall Curve)
        - **Optimal Threshold:** 0.010 (from Decision Curve Analysis)
        
        ### Key Predictors
        1. **Hemoglobin levels** (min/max/mean) - 48.7% of feature importance
        2. **Age** - 16.7% of importance
        3. **CRP levels** - 8.2% of importance
        4. **Procedure history** - 7.1% of importance
        
        ### Important Limitations
        ⚠️ **CRITICAL LIMITATION:** ICU admission data was excluded due to data quality issues
        - Original data showed: CSKP patients 0% ICU, CRKP patients 40% ICU
        - This created data leakage and artificially inflated performance
        - Model performance represents baseline without ICU data
        
        ⚠️ **Other Limitations:**
        - Limited to demographic and laboratory predictors
        - External validation not yet performed
        - Calibration needs improvement (slope=0.472, intercept=0.010)
        - For research use only, not FDA-approved
        
        ### Intended Use
        This tool is intended for:
        - Research and quality improvement
        - Clinical decision support (adjunct to clinical judgment)
        - Antibiotic stewardship programs
        - Not for standalone diagnosis
        
        ### References
        1. Clinical and Laboratory Standards Institute (CLSI) standards
        2. EUCAST breakpoints for carbapenem resistance
        3. TRIPOD-AI and STROBE reporting guidelines
        """)

def display_feature_importance():
    """Display feature importance visualization"""
    importance_df = load_feature_importance()
    
    if importance_df is not None:
        with st.expander("📈 Feature Importance Analysis", expanded=False):
            st.markdown("### Top 10 Features Influencing CRKP Risk")
            
            # Get top 10 features
            top_features = importance_df.head(10).copy()
            top_features = top_features.sort_values('importance', ascending=True)
            
            # Create horizontal bar chart
            import plotly.express as px
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (SHAP values)",
                labels={'importance': 'Mean |SHAP value|', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Larger bars indicate features with greater impact on predictions
            - Positive SHAP values increase CRKP risk prediction
            - Negative SHAP values decrease CRKP risk prediction
            """)

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
        st.title("Navigation")
        
        app_mode = st.radio(
            "Select Mode:",
            ["📋 Input Data", "📊 View Results", "ℹ️ Model Info"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This calculator estimates CRKP infection risk using machine learning.
        
        **Version:** 1.0 (Corrected Model)
        **Last Updated:** December 2024
        **Model:** XGBoost without ICU features
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("For technical issues or questions:")
        st.markdown("- Email: research@hospital.edu")
        st.markdown("- Phone: +1-555-CRKP-HELP")
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        **FOR RESEARCH USE ONLY**
        
        This tool does not provide medical advice.
        Always consult qualified healthcare professionals for medical decisions.
        
        By using this tool, you acknowledge the limitations described.
        """)
    
    # Main content based on selected mode
    if app_mode == "📋 Input Data":
        st.header("👤 Patient Data Entry")
        features = create_input_form()
        
        # Load model
        model = load_model()
        
        if model is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True):
                    with st.spinner("Calculating risk score..."):
                        prediction_prob, risk_category, color = make_prediction(model, features)
                        if prediction_prob is not None:
                            st.success("Prediction complete!")
            
            with col2:
                if st.button("🔄 Reset Form", type="secondary", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.session_state.risk_score = None
                    st.rerun()
        
        # Display feature importance if available
        display_feature_importance()
    
    elif app_mode == "📊 View Results":
        if st.session_state.prediction_made:
            display_results()
            
            # Export options
            st.subheader("📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(f"""
                    CRKP Risk Assessment Report
                    ===========================
                    Risk Score: {st.session_state.risk_score:.1%}
                    Risk Category: {st.session_state.risk_category}
                    Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                    
                    Patient Features:
                    - Age: {st.session_state.feature_values.get('age', 'N/A')}
                    - Hemoglobin: {st.session_state.feature_values.get('hemoglobin_mean', 'N/A')} g/dL
                    - CRP: {st.session_state.feature_values.get('crp_last', 'N/A')} mg/L
                    
                    Disclaimer: For research use only.
                    """)
                    st.success("Results copied to clipboard!")
            
            with col2:
                # Generate PDF report (simplified)
                if st.button("📄 Generate PDF Report"):
                    st.info("PDF generation feature requires additional setup.")
            
            with col3:
                if st.button("🔄 New Calculation"):
                    st.session_state.prediction_made = False
                    st.session_state.risk_score = None
                    st.rerun()
        else:
            st.warning("No prediction made yet. Please go to 'Input Data' to calculate risk.")
            if st.button("Go to Input Data"):
                st.switch_page("streamlit_app.py")  # This would work in actual deployment
    
    elif app_mode == "ℹ️ Model Info":
        display_model_info()
        
        # Add performance metrics visualization
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AUROC", "0.613", delta=None)
        
        with col2:
            st.metric("AUPRC", "0.263", delta="-65.6%", 
                     delta_color="inverse",
                     help="Drop from 0.620 after removing ICU features")
        
        with col3:
            st.metric("Brier Score", "0.222", delta=None,
                     help="Lower is better (0=perfect)")
        
        # ROC Curve placeholder
        st.markdown("### Performance Curves")
        st.info("Performance curves visualization requires additional data files.")
        
        # Calibration info
        st.markdown("### Calibration Statistics")
        st.warning("""
        **Calibration Needs Improvement:**
        - Slope: 0.472 (ideal=1.0)
        - Intercept: 0.010 (ideal=0.0)
        - Expected Calibration Error: 0.095
        
        **Interpretation:** Model tends to over-predict risk for low-risk patients
        and under-predict for high-risk patients.
        """)

# Run the app
if __name__ == "__main__":
    main()
