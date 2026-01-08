import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Load Model (Pipeline)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_telcochurn_pipeline.pkl")

model = load_model()

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📉 Telco Customer Churn Prediction")
st.write("Simulasikan kemungkinan customer melakukan churn berdasarkan karakteristik utama.")

st.divider()

# =====================================================
# User Inputs (CORE FEATURES ONLY)
# =====================================================
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=0.0
)

st.divider()

# =====================================================
# Auto-fill LOGIC (IMPORTANT)
# =====================================================
if internet_service == "No":
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"
else:
    online_security = "No"
    online_backup = "No"
    device_protection = "No"
    tech_support = "No"
    streaming_tv = "No"
    streaming_movies = "No"

# =====================================================
# Build Input DataFrame (FULL SCHEMA)
# =====================================================
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges
}])

# =====================================================
# Prediction
# =====================================================
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to **CHURN**")
        st.write(f"📊 Churn Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ Customer is likely to **STAY**")
        st.write(f"📊 Churn Probability: **{probability:.2%}**")

    st.caption(
        "⚠️ Prediction is based on model simulation with assumed default values for non-core features."
    )