import streamlit as st
import numpy as np
import pandas as pd 
import os 
 
current_dir = os.path.dirname(__file__) # Gets path 
file_path = os.path.join(current_dir, "march_2025_predictions_postcode.csv")


# Load the LLM predictions csv
@st.cache_data
def load_predictions():
    df = pd.read_csv(file_path)
    return df

predictions_df = load_predictions()


st.title("Burglary Risk Calculator")

# User inputs
# gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
postcode = st.text_input("Enter Postcode").strip().upper() # Strip and upper remove any leading/trailing white spaces and upper converts to upper case


if st.button("Predict Risk"):

    if not postcode:
        st.warning("Please enter a valid postcode")
    else: 
        # Try to match the postcode
        matched_row = predictions_df[predictions_df['Postcode'] == postcode]

        if not matched_row.empty:
            predicted_prob = matched_row.iloc[0]['Burglary_Probability']
            st.success(f"Predicted probability of burglary risk: {predicted_prob} ")

        else:
            st.error("No prediction found for entered postcode. Please check for typos")


    
