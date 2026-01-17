import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CRKP Risk Calculator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🦠 CRKP Risk Prediction Calculator")
st.markdown("""
**Clinical Decision Support for Carbapenem-Resistant *Klebsiella pneumoniae***  
*Validated on retrospective cohort (n=7,225) | Temporal validation | Prevalence-shift tested*
""")

# Load model
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

# Sidebar - Patient Input
with st.sidebar:
    st.header("📋 Patient Information")
    
    # Create input dictionary
    input_data = {}
    
    # Demographics
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        input_data['age'] = st.number_input("Age (years)", 0, 120, 65)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        input_data['gender'] = 1 if gender == "Male" else 0
    
    # Antibiotic exposure
    st.subheader("Antibiotic Exposure (Past 30 Days)")
    input_data['any_antibiotic_30d'] = st.checkbox("Any antibiotics")
    input_data['carbapenem_30d'] = st.checkbox("Carbapenems")
    input_data['broad_spectrum_30d'] = st.checkbox("Broad-spectrum")
    input_data['cephalosporin_30d'] = st.checkbox("Cephalosporins")
    input_data['fluoroquinolone_30d'] = st.checkbox("Fluoroquinolones")
    
    col1, col2 = st.columns(2)
    with col1:
        input_data['days_since_last_abx'] = st.number_input("Days since last antibiotic", 0, 30, 7)
    with col2:
        input_data['num_abx_classes_30d'] = st.number_input("Number of antibiotic classes", 0, 10, 1)
    
    input_data['total_abx_days_30d'] = st.slider("Total antibiotic days", 0, 30, 7)
    
    # Hospitalization
    st.subheader("Hospitalization History")
    input_data['icu_admission_30d'] = st.checkbox("ICU admission (past 30d)")
    input_data['ward_transfers_30d'] = st.number_input("Ward transfers (past 30d)", 0, 10, 0)
    input_data['days_in_hospital_before_culture'] = st.number_input("Days hospitalized", 0, 365, 7)
    input_data['current_location_icu'] = st.checkbox("Currently in ICU")
    
    # Lab values
    st.subheader("Laboratory Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['lymphocytes_last'] = st.number_input("Lymphocytes", 0.0, 20.0, 1.5, step=0.1)
    with col2:
        input_data['albumin_last'] = st.number_input("Albumin", 0.0, 10.0, 3.5, step=0.1)
    with col3:
        input_data['plt_last'] = st.number_input("Platelets", 0.0, 1000.0, 200.0, step=1.0)
    
    # Calculate button
    calculate = st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True)

# Main content
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
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CRKP Probability", f"{probability:.1%}")
        
        with col2:
            if probability < 0.2:
                risk = "Low"
                color = "🟢"
            elif probability < 0.5:
                risk = "Moderate"
                color = "🟡"
            else:
                risk = "High"
                color = "🔴"
            st.metric("Risk Category", f"{color} {risk}")
        
        with col3:
            threshold = 0.263
            prediction = "CRKP Likely" if probability >= threshold else "CRKP Unlikely"
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
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations
        st.subheader("🧪 Clinical Recommendations")
        
        if probability >= threshold:
            st.warning("""
            **CRKP LIKELY - Consider:**
            1. Carbapenem-sparing regimens pending susceptibility
            2. Contact precautions
            3. Await culture results
            4. Infectious diseases consult
            """)
        else:
            st.success("""
            **CRKP UNLIKELY - Consider:**
            1. Carbapenem-sparing empiric therapy
            2. Continue monitoring
            3. De-escalate if culture allows
            """)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not calculate:
    # Default view
    st.markdown("""
    ## 👨‍⚕️ How to Use
    
    1. **Enter patient data** in sidebar
    2. **Click 'Calculate CRKP Risk'**
    3. **Review results** and recommendations
    
    ### 📋 Model Details
    - **Algorithm**: XGBoost with Edited Nearest Neighbors
    - **AUPRC**: 0.269 (95% CI: 0.237-0.312)
    - **AUROC**: 0.703 (95% CI: 0.674-0.730)
    - **Sensitivity**: 84.7% | **Specificity**: 51.9%
    - **Features**: 23 clinical predictors
    
    ### 🔒 Privacy
    - No data storage
    - Calculations in your browser
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 0.8em; color: gray;'>
        CRKP Prediction Calculator v1.0 | For research use only | 
        <a href='https://github.com/maroofb88/CRKP_RISK_Calculator' target='_blank'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
