import streamlit as st

st.set_page_config(page_title="London Burglary Predictor", layout="centered")
st.title("🔐 London Burglary Predictor")

st.markdown("""
Welcome to the **London Burglary Risk Prediction Tool**, a data-driven platform designed to forecast the likelihood of residential burglary across London neighbourhoods.

---

### Why Use This Tool?

- **For Citizens:** Understand your area's burglary risk and take informed safety measures.
- **For Police:** Help allocate patrol resources efficiently based on predicted burglary hotspots.
- **For Policymakers:** Gain insights to support targeted crime reduction strategies.

---

Use the sidebar to navigate through different pages for interactive tools, police insights, ethics considerations, and safety recommendations.

---

Stay informed, stay safe!
""")

st.markdown("---")

st.markdown("**Enter your LSOA code**")
# Input field for LSOA
lsoa_input = st.text_input(label = "Your LSOA code is a 9 digit sequence (eg. E01000001):", placeholder = "Type your LSOA here...", max_chars = 9)

# Just a check to show what the user typed (can remove this if we wish to)
if lsoa_input:
    st.success(f"You Entered: {lsoa_input}")

st.markdown("---")
st.caption("Developed by the London Crime Data Analytics Team")

