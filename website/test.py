import streamlit as st
import pandas as pd
import numpy as np

st.title("🎉 Python Setup Test")
st.write("If you can see this, your setup is working!")

# Test data manipulation
data = pd.DataFrame({
    'numbers': [1, 2, 3, 4, 5],
    'squares': [1, 4, 9, 16, 25]
})

st.write("Here's a test dataframe:")
st.dataframe(data)

st.write("🎯 Your Python environment is ready for the burglary risk calculator!")