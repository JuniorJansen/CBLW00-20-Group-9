import streamlit as st
import numpy as np

st.title("Burglary Risk Calculator")

# User inputs
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
postcode = st.text_input("Enter Postcode")

if st.button("Predict Risk"):
    # Dummy encoding (just as example)
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    gender_encoded = gender_map.get(gender, 2)
    
    # Dummy postcode encoding (simple hash mod)
    postcode_encoded = hash(postcode) % 1000 if postcode else 0
    
    # Dummy calculation of probability (just for demo)
    dummy_prob = (0.3 * gender_encoded + 0.05 * age + 0.0001 * postcode_encoded) % 1
    
    st.success(f"Predicted probability of burglary risk: {dummy_prob:.2%}")
