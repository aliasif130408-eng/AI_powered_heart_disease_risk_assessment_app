import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title("AI Powered Heart Disease Risk Assessment App")

# Paths
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "heart.csv")
model_path = os.path.join(BASE_DIR, "best_model.pkl")

# Try to load model
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except:
    st.warning("Model file not found. Training a new model...")

    # Load Kaggle CSV
    data = pd.read_csv(csv_path)

    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_path)
    st.success("Model trained and saved!")

# Sidebar inputs
st.sidebar.header("Input Your Details")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", value=120)
    chol = st.sidebar.number_input("Serum Cholesterol", value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG results", [0, 1, 2])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.number_input("ST depression induced by exercise", value=1.0)
    slope = st.sidebar.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of major vessels colored", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Heart Disease Risk: ", "Yes" if prediction[0] == 1 else "No")

st.subheader("Prediction Probability")
st.write(prediction_proba)
