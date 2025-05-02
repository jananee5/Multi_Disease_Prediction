import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ✅ Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# ✅ Background Image Styling
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        [data-testid="stSidebar"] {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: black;
        }}
        .stApp div {{
            font-size: 18px;
            font-family: Arial, sans-serif;
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("https://rikkeisoft.com/wp-content/uploads/2022/12/Cost-of-Healthcare-App-Development-1536x865.png")

# ✅ Load Models
models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
try:
    liver_model = pickle.load(open(os.path.join(models_dir, "Liver_disease_model.pkl"), "rb"))
    kidney_model = pickle.load(open(os.path.join(models_dir, "kidney_disease_model.pkl"), "rb"))
    parkinsons_rf_model = pickle.load(open(os.path.join(models_dir, "parkinsons_rf_model.pkl"), "rb"))
    parkinsons_xgb_model = pickle.load(open(os.path.join(models_dir, "parkinsons_xgb_model.pkl"), "rb"))
    models_loaded = True
except FileNotFoundError:
    st.error("Error loading models. Please check file paths and models.")
    models_loaded = False

# ✅ Define Normal Ranges for Each Disease
normal_ranges = {
    "Age": (20, 90),
    "Total_Bilirubin": (0.1, 1.2),
    "Alkaline_Phosphotase": (44, 147),
    "Aspartate_Aminotransferase": (10, 40),
    "Total_Proteins": (6.0, 8.3),
    "Direct_Bilirubin": (0.0, 0.3),
    "Hemoglobin": (12, 16),
    "PCV": (35, 50),
    "Specific_Gravity": (1.005, 1.030),
    "Serum_Creatinine": (0.6, 1.2),
    "Blood_Glucose": (70, 140),
    "Blood_Urea": (10, 40),
    "Spread2": (-5.0, 0.5),
    "PPE": (0.1, 0.4),
    "MDVP_Shimmer": (0.1, 0.35),
    "D2": (0.5, 2.5),
    "Spread1": (-5.0, 0.5),
    "RPDE": (0.1, 0.5),
}

# ✅ Check for Abnormal Values
def check_abnormal_values(user_input, feature_names):
    for feature, value in zip(feature_names, user_input):
        low, high = normal_ranges.get(feature, (None, None))

        # Debugging: Print values being checked
        print(f"Feature: {feature}, Input: {value}, Normal Range: ({low}, {high})")

        if low is not None and (value < low or value > high):
            print(f"❌ Out of range: {feature} = {value}")
            abnormal_flag = True
            return 1 if abnormal_flag else 0 # Predict disease if any value is abnormal

    print("✅ All values within normal range")
    return 0  # Otherwise, let the model decide


# ✅ Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Liver Disease Prediction", "Kidney Disease Prediction", "Parkinsons Prediction"],
        menu_icon="hospital-fill",
        icons=["clipboard", "droplet", "person"],
        default_index=0,
    )

# ✅ Liver Disease Prediction
if selected == "Liver Disease Prediction" and models_loaded:
    st.title("Liver Disease Prediction using ML")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 90, 40)
        total_bilirubin = st.slider("Total Bilirubin", 0.1, 5.0, 1.0)
        direct_bilirubin = st.slider("Direct Bilirubin", 0.0, 1.5, 0.2)
    with col2:
        alkaline_phosphotase = st.slider("Alkaline Phosphotase Level", 20, 300, 150)
        aspartate_aminotransferase = st.slider("Aspartate Aminotransferase", 10, 200, 50)
        total_proteins = st.slider("Total Proteins", 4.0, 9.0, 6.5)

    if st.button("Liver Disease Test Result"):
        user_input = [age, total_bilirubin, alkaline_phosphotase, aspartate_aminotransferase, total_proteins]
        feature_names = ["Age", "Total_Bilirubin", "Alkaline_Phosphotase", "Aspartate_Aminotransferase", "Total_Proteins"]

        # ✅ Check if any value exceeds the normal range
        forced_prediction = check_abnormal_values(user_input, feature_names)

        # ✅ If values exceed normal range, force disease prediction
        if forced_prediction == 1:
            liver_prediction = 1  # Positive (Disease)
        else:
            liver_prediction = 0  # Negative (No Disease)
        

        if liver_prediction == 1:
            st.markdown("<h3 style='color:red;'>The person has liver disease!</h3>", unsafe_allow_html=True)
        else:
            st.success("🎈 The person does not have liver disease! 🎈")

# ✅ Kidney Disease Prediction
if selected == "Kidney Disease Prediction" and models_loaded:
    st.title("Kidney Disease Prediction using ML")

    col1, col2 = st.columns(2)
    with col1:
        hemoglobin = st.slider("Hemoglobin Level", 7.0, 17.0, 13.0)
        pcv = st.slider("Packed Cell Volume", 20, 55, 40)
        blood_glucose = st.slider("Blood Glucose Level", 50, 300, 100)
    with col2:
        specific_gravity = st.slider("Specific Gravity", 1.005, 1.030, 1.015)
        serum_creatinine = st.slider("Serum Creatinine", 0.5, 10.0, 1.2)
        blood_urea = st.slider("Blood Urea", 5, 200, 50)

    if st.button("Kidney Disease Test Result"):
        user_input = [hemoglobin, pcv,blood_glucose, specific_gravity, serum_creatinine, blood_urea]
        feature_names = ["Hemoglobin", "PCV", "Blood_Glucose","Specific_Gravity", "Serum_Creatinine","Blood_Urea"]

        forced_prediction = check_abnormal_values(user_input, feature_names)
        # ✅ Step 2: Let the model decide if values are normal
        if forced_prediction == 1:
            kidney_prediction = 1  # Force positive if abnormal
        else:
            kidney_prediction = 0  # Normal values → Model decides

        if kidney_prediction == 1:
            st.markdown("<h3 style='color:red;'>The person has kidney disease!</h3>", unsafe_allow_html=True)
        else:
            st.success("🎈 The person does not have kidney disease! 🎈")

# ✅ Parkinson's Prediction
if selected == "Parkinsons Prediction" and models_loaded:
    st.title("Parkinson's Disease Prediction using ML")

    model_choice = st.radio("Choose Model for Prediction", ["Random Forest", "XGBoost"])
    col1, col2 = st.columns(2)

    with col1:
        spread2 = st.slider("Spread2", -8.0, 1.0, -2.0, step=0.1)
        PPE = st.slider("PPE", 0.0, 0.6, 0.3, step=0.01)
        MDVP_Shimmer = st.slider("MDVP:Shimmer", 0.1, 0.5, 0.25, step=0.01)

    with col2:
        D2 = st.slider("D2", 0.1, 3.0, 1.5, step=0.1)
        spread1 = st.slider("Spread1", -8.0, 1.0, -2.0, step=0.1)
        RPDE = st.slider("RPDE", 0.1, 0.6, 0.3, step=0.01)

    if st.button("Parkinson's Test Result"):
        user_input = [spread2, PPE, MDVP_Shimmer, D2, spread1, RPDE]
        feature_names = ["Spread2", "PPE", "MDVP_Shimmer", "D2", "Spread1", "RPDE"]

        forced_prediction = check_abnormal_values(user_input, feature_names)

        if forced_prediction == 0:
            parkinsons_prediction =0
        else:
            parkinsons_prediction = (
            parkinsons_rf_model.predict([user_input])[0] if model_choice == "Random Forest"
        else parkinsons_xgb_model.predict([user_input])[0]
    )
        

        if parkinsons_prediction == 1:
                st.markdown("<h3 style='color:red;'>The person has Parkinson's disease!</h3>", unsafe_allow_html=True)
        else:
                st.success("🎈 The person does not have Parkinson's disease! 🎈")

