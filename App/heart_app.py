import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #e6f0ff;  /* Light blue background */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    /* Button color */
    div.stButton > button:first-child {
        background-color: #3399ff;  /* Bright blue */
        color: white;
    }
    div.stButton > button:hover {
        background-color: #0066cc;  /* Darker blue on hover */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
st.title("💓 AI-Powered Heart Disease Risk Assessment App")
st.write("Answer the following questions to assess your risk of heart disease.")

# --- USER INPUTS ---
age = st.number_input("Age", min_value=1, max_value=120, step=1)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type (cp)",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

trestbps = st.number_input("Resting Blood Pressure (trestbps) in mm Hg", min_value=50, max_value=250, step=1)
chol = st.number_input("Serum Cholestoral (chol) in mg/dl", min_value=100, max_value=600, step=1)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])

restecg = st.selectbox(
    "Resting Electrocardiographic Results (restecg)",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, step=1)
exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)

slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", ["Normal", "Fixed Defect", "Reversible Defect"])

# --- CONVERT INPUTS TO NUMERIC FORM ---
sex_map = {"Male": 0, "Female": 1}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"No": 0, "Yes": 1}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# Create input DataFrame
input_data = {
    'age': age,
    'sex': sex_map[sex],
    'cp': cp_map[cp],
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_map[fbs],
    'restecg': restecg_map[restecg],
    'thalach': thalach,
    'exang': exang_map[exang],
    'oldpeak': oldpeak,
    'slope': slope_map[slope],
    'ca': ca,
    'thal': thal_map[thal]
}

input_df = pd.DataFrame([input_data])

# --- PREDICTION ---
if st.button("🔍 Predict Heart Disease Risk"):
    try:
        prediction = model.predict(input_df)
        result = "🩺 **High Risk of Heart Disease**" if prediction[0] == 1 else "💖 **Low Risk of Heart Disease**"
        st.subheader(result)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Input DataFrame:", input_df)



