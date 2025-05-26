import streamlit as st

# Page setup
st.set_page_config(page_title="London Burglary Predictor", layout="centered", page_icon="🔐")

# Header
st.title("🔐 London Burglary Risk Predictor")
st.subheader("Estimate your LSOA’s burglary risk using data and machine learning.")

st.markdown("---")

# Section: Who is this for and What does it do 
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👥 Who is this for?")
    st.write("""
    - **Citizens**: Check your local burglary risk and get safety suggestions.
    - **Police**: Allocate resources based on data-driven risk prediction.
    """)

with col2:
    st.markdown("### 📊 What does this tool do?")
    st.write("""
    - Uses a machine learning model trained on London LSOA residential burglary data.
    - Predicts burglary risk based on your local area and demographics.
    - Offers advice and insights if your risk is above a certain threshold.
    """)


st.markdown("---")

# Navigation info
st.markdown("ℹ️ Navigate using the sidebar to access calculator, insights, model information and transparency, and safety advice.")

# Footer
st.caption("Developed for 4CBLW00-20 - Group 9")

