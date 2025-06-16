import streamlit as st
import os

st.set_page_config(page_title="Ethics & Model Info", layout="centered")
st.title("🔍 Ethics & Model Transparency")

st.markdown("""
<style>

/* Main background and base font color */
.stApp {
    background-color: #eafafa !important;
    color: #000000 !important;
}

/* Text styling for markdown and headings */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #000000 !important;
}

/* Inputs */
input[type="text"], textarea, .stTextInput > div > input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
}

/* Placeholder text */
input::placeholder, textarea::placeholder {
    color: #444 !important;
    opacity: 1 !important;
}

/* Buttons */
button {
    background-color: #007acc !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}

button:hover {
    background-color: #005fa3 !important;
}

/* Metric labels and values */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #000000 !important;
}

/* Alerts */
.stAlert {
    color: #000000 !important;
}

/* Hide theme toggle */
[data-testid="theme-toggle"] {
    display: none !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #1e1e1e !important;
    color: white !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ───── Expander box fix ───── */
div[data-testid="stExpander"] {
    border: 1px solid #999 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

div[data-testid="stExpander"] > details > summary {
    font-weight: bold !important;
    color: #000000 !important;
}

div[data-testid="stExpander"] > details {
    background-color: #f9f9f9 !important;
}

</style>
""", unsafe_allow_html=True)

found_paths = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.lower() == "imd.png":
            found_paths.append(os.path.join(root, file))

if found_paths:
    st.success(f"Found IMD.png at:\n\n{found_paths}")
    # Optionally display the first one found
    st.image(found_paths[0], caption="Auto-detected IMD.png", use_container_width=True)
else:
    st.error("❌ 'IMD.png' not found in any subdirectory.")


st.markdown("---")

# IMD Scores Explanation
with st.expander("📊 What Are IMD Scores?"):
    st.markdown("""
    The **Index of Multiple Deprivation (IMD)** is the UK government's official measure of relative deprivation across small areas (LSOAs).
    
    It combines **seven domains**:
    - **Income Deprivation**
    - **Employment Deprivation**
    - **Education, Skills and Training**
    - **Health Deprivation and Disability**
    - **Crime**
    - **Barriers to Housing and Services**
    - **Living Environment**
    
    Each area is ranked from **1 (most deprived)** to **32,844 (least deprived)**.  
    We incorporate IMD data to help identify communities that may benefit most from enhanced burglary prevention efforts.
    """)

    with st.container():
        st.image(
            "pages/IMD.png", 
            caption="IMD Domains and Their Indicators", 
            use_container_width=True
        )

    st.markdown("""
    **If you want to learn more about the IMDs:**
    - [IMD Explorer](https://www.gov.uk/guidance/english-indices-of-deprivation-2019-mapping-resources#indices-of-deprivation-2019-explorer-postcode-mapper)  
    - [Technical Report](https://assets.publishing.service.gov.uk/media/5d8b387740f0b609909b5908/IoD2019_Technical_Report.pdf)
    """)




# Custom IMD Features
with st.expander("🧩 Our Custom Deprivation Index"):
    st.markdown("""
To capture burglary risk more accurately, we developed a **custom deprivation index** using additional factors inspired by academic literature and our own analysis. 
We deemed found these factors to be excluded from the orignal IMD analysis, but taking into account the rapid change of our society, we deemed them
valuable indicators of deprivation within London. The new factors include:

                
## Digital Exclusion
- **Effect on Burglaries**: Reduced online access lowers reporting and awareness.
- **Takes into account per LSOA**:
    - Internet Access Quality
    - Digital Literacy Rates
    - Device Access per Household


## Climate Vulnerability and Energy Poverty
- **Effect on Burglaries**: Poor building conditions make homes easier targets.
- **Takes into account per LSOA**:
    - Households in Engery Poverty (cannot affor heating/cooling)
    - Local Energy Efficiency (EPC ratings of buildings)
                
## Transport Accessibility
- **Effect on Burglaries**: High access = easier entry/escape for burglars  || Low access = fewer witnesses, higher success rate
- **Takes into account per LSOA**:
    - Walking time from the point-of interest to the public transport access points
    - The reliability of the service modes available
    - The number of services available within the catchment
    - The level of service at the public transport access points - eg. average waiting time

## Age Demographics
- **Effect on Burglaries**: Young adult–dominated areas may face higher risk based on literature read.

These were used alongside IMD data to enhance predictive performance and maintain relevance for modern policing.
                
Additionally, we were thinking about adding Mental Health and Social Isolation as a factor to our custom deprivation index. However, we found the datasets
were more relevant for the offender (burglar) rather than the victim of the crime. Thus, we decided to exclude it as a factor as out goal was to predict who the victims would be.
                
If you want to learn more about these factors and the data used:
- [Transport Accessibility](https://data.london.gov.uk/dataset/public-transport-accessibility-levels#:~:text=The%20method%20is%20essentially%20a,excellent%20access%20to%20public%20transport.)
- [Climate Vulnerability and Energy Poverty](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/medianenergyefficiencyscoreenglandandwales)
- [Digital Exclusion](https://www.ons.gov.uk/peoplepopulationandcommunity/householdcharacteristics/homeinternetandsocialmediausage/articles/digitalpropensityindexforenglandandwaleslsoas/census2021)
- [Age](https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/articles/overviewofburglaryandotherhouseholdtheft/englandandwales?utm_source=chatgpt.com)
""")

with st.expander("🤖 How Our Model Works"):
    st.markdown("""
We use a **two-model Random Forest approach** to predict burglary risk across London. This helps us understand both the likelihood of a burglary happening and how many burglaries we expect to occur in specific areas in the following month.

- **Random Forest Regression**: Estimates the expected *number* of burglaries for each area (LSOA).
- **Random Forest Classification**: Predicts the *probability* that a burglary will happen in a given area.

---

### 🔍 Why Random Forest?

In our search for the best prediciton outcome, we tested of models:

- **Random Forest**
- **XGBoost**
- **Gradient Boosting**
- **LightGBM**

After testing these models, we decided to use **Random Forest**:

- Strong performance on both prediction tasks  
- Robust to overfitting (doesn’t memorize the training data)  
- Clear insights into which factors matter most  
- Allows us to adjust the model to focus more on safety

---

### 🚨 Built-In Ethical Safeguards

We also introduced an ethical safeguard by adjusting the model to favor overestimating burglary risks slightly.

> **Why?** It’s better to **overestimate risk** and allocate resources preemptively than to **underestimate** and leave areas vulnerable.

This decision aligns with our priority of public safety and preventive policing.
""")

# Privacy & Ethics
with st.expander("🔐 Privacy & Ethical Considerations"):
    st.markdown("""
    Ethical integrity and user privacy are foundational to our work.

    ### ✅ Data Privacy
    - No personal or individual-level data is used.
    - Predictions are made only at the neighborhood (LSOA) level to avoid identifying individuals or properties.
    - We have used the London's Police anonymisation procedure to ensure nobody's privacy is violated. The latitude and longitude locations of incidents reported always represent the approximate location of a crime — not the exact place that it happened.
    - Data is processed and stored securely. No personally identifiable information is ever retained.
    - [Read more about anonymisation here.](https://data.police.uk/about/#anonymisation)

    ### ✅ Transparency
    - This page provides full visibility into our data sources, databases and modeling techniques.
    - We explain both the capabilities **and** the limitations of our model.
    - Users can explore how various factors influence local risk scores.

    ### ✅ Fairness & Accountability
    - We acknowledge the potential for area-based models to stigmatize communities. However, our objective is **not to rank individuals**, but to support informed, equitable prevention strategies at a local level.
    - We added new domains like **Digital Exclusion**, **Transport Access**, and **Climate Vulnerability** to reflect real-world risks, not enforcement patterns.

    ### ✅ Intended Use & Misuse Prevention
    - Predictions are not absolute — they represent **probabilities**, not guarantees.
    - We do not label areas as "dangerous". Instead, we highlight relative ris and suggest appropriate prevention strategies.
    - Risk scores are always shown alongside safety tips, giving residents proactive tools, not fear.
    """)


