import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import sys
import os

# Add 'src' to python path so we can import our FinancialEngine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from financial_engine import FinancialEngine 

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AgriGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. LOAD RESOURCES (Cached for Speed) ---
@st.cache_resource
def load_model():
    # 1. Define the exact path to your trained model
    # Windows paths use backslashes (\), Python prefers forward slashes (/)
    # Update this string to match YOUR specific location if different
    model_path = 'runs/classify/weights/agriguard_model/weights/best.pt' 

    # 2. Check if it exists
    if os.path.exists(model_path):
        st.success(f"✅ Custom AgriGuard Model Loaded: {model_path}")
        return YOLO(model_path)
    else:
        st.error(f"❌ Custom Model NOT Found at: {model_path}")
        st.warning("Please check the 'weights' folder. Did training finish?")
        # STOP the app here so you don't get 'butternut squash' results
        st.stop()

@st.cache_resource
def load_financial_engine():
    # We need to import the class from the src file
    # To avoid import errors with numbers in filenames, we use importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("FinancialEngine", "src/financial_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FinancialEngine()

model = load_model()
fin_engine = load_financial_engine()

# --- 2. TREATMENT DATABASE (Static Knowledge Base) ---
treatments = {
    'Blight': "Apply fungicides containing mancozeb or chlorothalonil. Improve air circulation.",
    'Mildew': "Use sulfur-based sprays or neem oil. Remove infected leaves immediately.",
    'Rust': "Apply copper-based fungicides. Rotate crops to prevent recurrence.",
    'Spot': "Avoid overhead watering. Use copper soap or certified organic fungicides.",
    'Mite': "Spray water to knock them off. Introduce predatory mites or use neem oil.",
    'Healthy': "Crop is healthy! Continue standard irrigation and monitoring."
}

def get_treatment(disease_name):
    for key, val in treatments.items():
        if key.lower() in disease_name.lower():
            return val
    return "Consult a local agronomist for specific chemical controls."

# --- 3. UI LAYOUT ---
st.title("🌾 AgriGuard AI: Risk & Credit Engine")
st.markdown("### Bridging the gap between *Crop Health* and *Financial Access*.")

# Sidebar
mode = st.sidebar.radio("Select User Mode:", ["👨‍🌾 Farmer (Diagnosis)", "🏦 Banker (Risk Analysis)"])

if mode == "👨‍🌾 Farmer (Diagnosis)":
    st.header("🍂 Plant Disease Diagnosis")
    uploaded_file = st.file_uploader("Upload a photo of your crop leaf:", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("### AI Diagnosis Results")
            if st.button("Analyze Leaf"):
                with st.spinner("Running YOLOv8 Vision Model..."):
                    # Run Inference
                    results = model(image)
                    
                    # Process Results (Classification)
                    top_result = results[0].probs.top1
                    conf = results[0].probs.top1conf.item()
                    class_name = results[0].names[top_result] # e.g., "Tomato___Early_blight"
                    
                    # Parse Name
                    clean_name = class_name.replace("___", " - ").replace("_", " ")
                    disease_only = class_name.split("___")[-1]
                    
                    # Display Metrics
                    st.metric(label="Detected Condition", value=clean_name)
                    st.metric(label="Model Confidence", value=f"{conf*100:.1f}%")
                    
                    # Display Treatment
                    st.info(f"💊 **Recommended Treatment:** {get_treatment(disease_only)}")
                    
                    # Store for Banker Mode (Session State)
                    st.session_state['last_diagnosis'] = class_name
                    st.success("Diagnosis saved to system.")

elif mode == "🏦 Banker (Risk Analysis)":
    st.header("📊 Micro-Credit Risk Assessment")
    
    # Input Form
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox("State", ["Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Tamil Nadu"])
            crop = st.selectbox("Crop", ["Maize", "Rice", "Potato", "Tomato", "Wheat"])
        with col2:
            land_area = st.number_input("Land Area (Acres)", min_value=1.0, value=5.0)
            
            # Auto-fill disease if coming from Farmer Tab
            default_disease = st.session_state.get('last_diagnosis', 'Healthy')
            disease_input = st.text_input("Disease Condition (Auto-filled from Vision Model)", value=default_disease)
            
        submit = st.form_submit_button("Generate Risk Report")
    
    if submit:
        # Run Financial Engine
        try:
            report = fin_engine.calculate_risk_profile(state, crop, disease_input, land_area)
            
            # Display Dashboard
            st.divider()
            st.subheader(f"Risk Profile: {report['Recommendation']}")
            
            # Key Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Credit Score", report['Credit_Eligibility_Score'], delta_color="normal")
            m2.metric("Yield Loss Risk", report['Yield_Loss_Pct'], delta_color="inverse")
            m3.metric("Proj. Revenue", f"₹{report['Projected_Revenue_INR']:,}")
            m4.metric("Revenue at Risk", f"₹{report['Revenue_at_Risk_INR']:,}", delta_color="inverse")
            
            # Detailed Table
            st.table(pd.DataFrame([report]).T.rename(columns={0: "Value"}))
            
            # Visual Logic
            if report['Credit_Eligibility_Score'] > 70:
                st.balloons()
            else:
                st.error("High Risk Detected. Mandatory Crop Insurance Recommended.")
                
        except Exception as e:
            st.error(f"Error in calculation: {e}")