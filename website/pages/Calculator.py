import streamlit as st
import numpy as np
import pandas as pd 
import os 
import math

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
file_path_pred = os.path.join(current_dir, "march_2025_predictions_postcode.csv")
file_path_households = os.path.join(current_dir, "Households.csv")

# Load the LLM predictions csv
@st.cache_data
def load_predictions(fname):
    df = pd.read_csv(fname)
    return df

predictions_df = load_predictions(file_path_pred)
households_df = load_predictions(file_path_households)

# Merging the two data sets on the common LSOA key 
perHousePredictions_df = pd.merge(predictions_df, households_df, left_on = 'LSOA code', right_on = 'Lower layer Super Output Areas Code', how = 'left')

perHousePredictions_df['Per House Risk'] = perHousePredictions_df['Burglary_Probability'] / perHousePredictions_df['Observation']

# Normalize the postcode column once (add a new column)
perHousePredictions_df['Postcode_Clean'] = perHousePredictions_df['Postcode'].str.replace(" ", "").str.upper()

st.title("Burglary Risk Calculator")


# User inputs
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)

postcode = st.text_input("Enter Postcode (e.g. EC2V 7DY, RM9 5PB)").strip()

if st.button("Predict Risk"):

    if not postcode:
        st.warning("🚨 Please enter a valid postcode.")

    else: 
        # Normalize user input postcode
        postcode_clean = postcode.replace(" ", "").upper()

        # Match normalized postcode in the dataframe
        matched_row = perHousePredictions_df[perHousePredictions_df['Postcode_Clean'] == postcode_clean]

        if not matched_row.empty:
            predicted_prob = matched_row.iloc[0]['Burglary_Probability']
            houseRisk_prob = matched_row.iloc[0]['Per House Risk']

            # Show per-house risk as a percentage with 4 decimal places
            per_house_risk_percentage = houseRisk_prob * 100
            st.metric(label="🔐 The risk of your house being burgled in the next month is", value=f"{per_house_risk_percentage:.4f}%")
            st.progress(float(per_house_risk_percentage))
            st.metric(label="🔐 Burglary Risk Probability in your LSOA in the next month", value=f"{predicted_prob:.2%}")
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


lsoa_code = "E01000013"
filtered_row = perHousePredictions_df[perHousePredictions_df['LSOA code'] == lsoa_code]



