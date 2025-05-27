import streamlit as st
import numpy as np
import pandas as pd 
import os 

st.markdown(
    """
    <style>
    .stApp {
        background-color: #eafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
current_dir = os.path.dirname(__file__)  # Gets path 
file_path = os.path.join(current_dir, "march_2025_predictions_postcode.csv")

# Load the LLM predictions csv
@st.cache_data
def load_predictions():
    df = pd.read_csv(file_path)
    return df

predictions_df = load_predictions()

# Normalize the postcode column once (add a new column)
predictions_df['Postcode_Clean'] = predictions_df['Postcode'].str.replace(" ", "").str.upper()

st.title("Burglary Risk Calculator")

# User inputs
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)

postcode = st.text_input("Enter Postcode (e.g., EC2V 7DY)").strip()

if st.button("Predict Risk"):

    if not postcode:
        st.warning("🚨 Please enter a valid postcode.")

    else: 
        # Normalize user input postcode
        postcode_clean = postcode.replace(" ", "").upper()

        # Match normalized postcode in the dataframe
        matched_row = predictions_df[predictions_df['Postcode_Clean'] == postcode_clean]

        if not matched_row.empty:
            predicted_prob = matched_row.iloc[0]['Burglary_Probability']
            
            # Convert probability to percentage (if not already)
            st.metric(label="🔐 Burglary Risk Probability", value=f"{predicted_prob:.2%}")
            st.progress(float(predicted_prob))

            # Risk level indicator
            if predicted_prob < 0.2:
                st.success("🟢 Low Risk Area")
            elif predicted_prob < 0.5:
                st.warning("🟡 Moderate Risk Area")
            else:
                st.error("🔴 High Risk Area – Take extra precautions, refer to the Safety Recommendations page")

        else:
            st.error("❌ No prediction found for entered postcode. Please check for typos.")
