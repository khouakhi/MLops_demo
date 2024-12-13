import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("v1/model_v1.pkl")

# Streamlit app title
st.title("Student Performance Prediction")

# Input form for school directors
st.header("Enter Student Attributes")

# Inputs for features
G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, step=1)
studytime = st.selectbox("Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", options=[1, 2, 3, 4])
famsup = st.radio("Family Support", options=["Yes", "No"])

# Encode famsup into two columns
famsup_yes = 1 if famsup == "Yes" else 0
famsup_no = 1 if famsup == "No" else 0

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        "G1": [G1],
        "studytime": [studytime],
        "famsup_no": [famsup_no],
        "famsup_yes": [famsup_yes]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")
