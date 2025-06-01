import streamlit as st

st.set_page_config(page_title="Ethics & Model Info", layout="centered")
st.title("🔍 Ethics & Model Transparency")

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
            "C:/Users/20231105/OneDrive/YEAR2/Q4/CBL/IMD.png", 
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

# Model Explanation
with st.expander("🤖 How Our Model Works"):
    st.markdown("""
We trained a **Random Forest Classifier** on monthly burglary data at the LSOA level, enhanced with our custom deprivation index databases.

### Why Random Forest?
- Handles complex interactions between variables
- Resistant to overfitting
- Provides interpretable feature importance scores

We also tested:
- Linear Regression  
- Ridge & Lasso (regularization)  
- Logistic Regression

But Random Forest offered the best trade-off between accuracy and interpretability.
""")

# Privacy & Ethics
with st.expander("🔐 Privacy & Ethical Considerations"):
    st.markdown("""
    Ethical integrity and user privacy are foundational to our work.

    ### ✅ Data Privacy
    - No personal or individual-level data is used.
    - Predictions are exclusively made at the neighborhood (LSOA) level.
    - We have used the London's Police anonymisation procedure to ensure nobody's privacy is violated. The latitude and longitude locations of incidents reported always represent the approximate location of a crime — not the exact place that it happened
    - [Read more about anonymisation here.](https://data.police.uk/about/#anonymisation)

    ### ✅ Transparency
    - This page provides full visibility into our data sources, databases and modeling techniques.
    - Users can explore how various factors influence local risk scores.

    ### ✅ Fairness & Accountability
    - We acknowledge the potential for area-based models to inadvertently stigmatize communities. However, our objective is **not to rank individuals**, but to support informed, equitable prevention strategies at the local level.
    """)


