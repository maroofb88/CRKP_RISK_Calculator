import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pathlib import Path
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
st.title("🦠 CRKP Risk Prediction Calculator")
st.markdown("""
**Clinical Decision Support for Carbapenem-Resistant *Klebsiella pneumoniae***  
*Validated on retrospective cohort (n=7,225) | Temporal validation | Prevalence-shift tested*
""")

# Load model and preprocessor
@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    try:
        model = joblib.load('models/xgb_enn_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        with open('models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        return model, preprocessor, feature_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, preprocessor, feature_info = load_model()

# Sidebar for model info
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("AUPRC", "0.269", delta=None)
    st.metric("AUROC", "0.703", delta=None)
    st.metric("Sensitivity", "84.7%", delta=None)
    st.metric("Specificity", "51.9%", delta=None)
    st.metric("Optimal Threshold", "0.263", delta=None)
    
    st.markdown("---")
    st.header("🎯 Clinical Utility")
    st.markdown("""
    **High NPV (94.2%):** Good for ruling out CRKP  
    **Moderate PPV (26.7%):** Confirmatory testing needed  
    **Best used as:** Screening tool for carbapenem-sparing therapy
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    **FOR RESEARCH USE ONLY**
    
    Not for direct clinical decision-making.
    Always use clinical judgment and confirmatory testing.
    """)

# Main app content
tab1, tab2, tab3 = st.tabs(["📋 Patient Input", "📊 Results", "ℹ️ Model Info"])

with tab1:
    st.header("Patient Information Input")
    st.markdown("All data must be available **before** culture order time.")
    
    input_data = {}
    
    # Demographics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Demographics")
        input_data['age'] = st.number_input("Age (years)", 0, 120, 65)
        gender = st.selectbox("Gender", ["Female", "Male"])
        input_data['gender'] = 1 if gender == "Male" else 0
    
    # Antibiotic exposure
    with col2:
        st.subheader("Antibiotic Exposure (Past 30 Days)")
        input_data['any_antibiotic_30d'] = st.checkbox("Any antibiotics")
        input_data['carbapenem_30d'] = st.checkbox("Carbapenems")
        input_data['broad_spectrum_30d'] = st.checkbox("Broad-spectrum")
        input_data['cephalosporin_30d'] = st.checkbox("Cephalosporins")
        input_data['fluoroquinolone_30d'] = st.checkbox("Fluoroquinolones")
    
    # Medication details
    col1, col2 = st.columns(2)
    with col1:
        input_data['days_since_last_abx'] = st.number_input("Days since last antibiotic", 0, 30, 7)
    with col2:
        input_data['num_abx_classes_30d'] = st.number_input("Number of antibiotic classes", 0, 10, 1)
    
    input_data['total_abx_days_30d'] = st.slider("Total antibiotic days", 0, 30, 7)
    
    # Hospitalization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hospitalization History")
        input_data['icu_admission_30d'] = st.checkbox("ICU admission (past 30d)")
        input_data['ward_transfers_30d'] = st.number_input("Ward transfers (past 30d)", 0, 10, 0)
    with col2:
        input_data['days_in_hospital_before_culture'] = st.number_input("Days hospitalized before culture", 0, 365, 7)
        input_data['current_location_icu'] = st.checkbox("Currently in ICU")
    
    # Laboratory values
    st.subheader("Laboratory Values (Most Recent)")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['lymphocytes_last'] = st.number_input("Lymphocytes (x10⁹/L)", 0.0, 20.0, 1.5, step=0.1)
    with col2:
        input_data['albumin_last'] = st.number_input("Albumin (g/dL)", 0.0, 10.0, 3.5, step=0.1)
    with col3:
        input_data['plt_last'] = st.number_input("Platelets (x10⁹/L)", 0.0, 1000.0, 200.0, step=1.0)
    
    # Calculate button
    calculate = st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True)

with tab2:
    st.header("Prediction Results")
    
    if 'probability' not in st.session_state:
        st.info("Enter patient data and click 'Calculate CRKP Risk' to see results")
    else:
        probability = st.session_state.probability
        risk_category = st.session_state.risk_category
        prediction = st.session_state.prediction
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CRKP Probability", f"{probability:.1%}")
        with col2:
            st.metric("Risk Category", risk_category)
        with col3:
            st.metric("Prediction", prediction)
        
        # Gauge chart
        st.subheader("Risk Visualization")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "CRKP Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "green"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 26.3  # 0.263 threshold
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations
        st.subheader("🧪 Clinical Recommendations")
        if probability >= 0.263:
            st.warning("""
            **CRKP LIKELY - Consider:**
            1. **Empirical therapy**: Consider carbapenem-sparing regimens pending susceptibility
            2. **Infection control**: Implement contact precautions
            3. **Diagnostics**: Await culture and sensitivity results
            4. **Consultation**: Consider infectious diseases consult
            """)
        else:
            st.success("""
            **CRKP UNLIKELY - Consider:**
            1. **Empirical therapy**: May consider carbapenem-sparing regimens
            2. **Monitoring**: Continue clinical monitoring
            3. **De-escalation**: De-escalate therapy if culture results allow
            """)

with tab3:
    st.header("Model Information & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance")
        st.markdown("""
        **Primary Metrics (Temporal Validation):**
        - **AUPRC**: 0.269 (95% CI: 0.237-0.312)
        - **AUROC**: 0.703 (95% CI: 0.674-0.730)
        - **Sensitivity**: 84.7% (80.5-89.1%)
        - **Specificity**: 51.9% (48.6-54.7%)
        - **PPV**: 26.7% (23.3-29.7%)
        - **NPV**: 94.2% (92.6-95.8%)
        - **Brier Score**: 0.256
        - **Optimal Threshold**: 0.263 (maximizing F1)
        """)
    
    with col2:
        st.subheader("🏗️ Model Architecture")
        st.markdown("""
        **Algorithm**: XGBoost with Edited Nearest Neighbors  
        **Features**: 23 clinical predictors  
        **Imbalance Strategy**: Edited Nearest Neighbors (best of 6 strategies)  
        **Validation**: 5-fold stratified CV + temporal validation  
        **Training**: 5,780 patients (12.7% CRKP)  
        **Testing**: 1,445 patients (17.2% CRKP)  
        **Temporal Split**: ≤2023-07-07 / ≥2023-07-07
        """)
    
    st.subheader("📊 Prevalence-Shift Analysis")
    st.markdown("""
    Model evaluated across 14 CR:CS ratios (1:1 to 1:50):
    - **AUPRC range**: 0.373 (50% prevalence) to 0.037 (2% prevalence)
    - **AUPRC degradation**: 90% at low prevalence
    - **AUROC stability**: SD = 0.001 across prevalence shifts
    """)
    
    st.subheader("🎯 Feature Importance (Top 10)")
    st.markdown("""
    1. **days_since_last_abx** - Most important predictor
    2. **Recent antibiotic exposure patterns**
    3. **ICU admission history**
    4. **Hospitalization duration**
    5. **Age categories**
    6. **Laboratory values** (lymphocytes, albumin, platelets)
    7. **Ward transfers**
    8. **Specific antibiotic classes**
    """)
    
    st.subheader("⚠️ Limitations")
    st.markdown("""
    1. **Retrospective design**: Subject to biases of observational data
    2. **Single center**: External validation needed
    3. **Missing data**: Some laboratory values had high missingness
    4. **Prevalence sensitivity**: Performance varies with local prevalence
    5. **Research tool**: Not yet prospectively validated
    """)

# Prediction logic
if calculate and model is not None:
    try:
        # Prepare input data
        age = input_data['age']
        input_data['age_lt_50'] = 1 if age < 50 else 0
        input_data['age_50_65'] = 1 if 50 <= age < 65 else 0
        input_data['age_65_80'] = 1 if 65 <= age < 80 else 0
        input_data['age_ge_80'] = 1 if age >= 80 else 0
        input_data['age_missing'] = 0
        input_data['gender_missing'] = 0
        
        # Create DataFrame
        feature_order = feature_info['feature_names']
        input_df = pd.DataFrame([input_data])
        
        for feat in feature_order:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        input_df = input_df[feature_order]
        
        # Predict
        X_processed = preprocessor.transform(input_df)
        probability = model.predict_proba(X_processed)[0, 1]
        
        # Determine results
        if probability < 0.2:
            risk_category = "Low"
        elif probability < 0.5:
            risk_category = "Moderate"
        else:
            risk_category = "High"
        
        threshold = 0.263
        prediction = "CRKP Likely" if probability >= threshold else "CRKP Unlikely"
        
        # Store in session state
        st.session_state.probability = probability
        st.session_state.risk_category = risk_category
        st.session_state.prediction = prediction
        
        # Switch to results tab
        st.switch_page("streamlit_app.py#tab2")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 0.8em; color: gray;'>
        CRKP Prediction Calculator v1.0 | For research use only | 
        <a href='https://github.com/maroofb88/CRKP_RISK_Calculator' target='_blank'>GitHub Repository</a> |
        Model ID: XGB_ENN_v1 | Validation: Temporal 80/20
    </p>
</div>
""", unsafe_allow_html=True)
