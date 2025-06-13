import streamlit as st

st.set_page_config(page_title="Safety Recommendations", layout="centered")

# Background styling
st.markdown(
    """
    <style>
    /* Force dark theme colors */
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    /* Apply dark mode to widgets */
    .css-1cpxqw2, .css-ffhzg2, .css-1y4p8pa {
        background-color: #262730 !important;
        color: #ffffff !important;
    }

    /* Hide Streamlit theme switcher UI */
    [data-testid="theme-toggle"] {
        display: none !important;
    }

    /* General text fix */
    .css-qrbaxs, .css-1d391kg {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ Safety Recommendations")
st.markdown("Protect yourself and your community with these strategies.")
st.markdown("[See the European Crime Prevention Guide for Preventing Domestic Burglaries](https://eucpn.org/sites/default/files/document/files/2106_The%20prevention%20of%20domestic%20burglary_LR.pdf#:~:text=night%20owing%20to%20the%20increased,9)")
st.markdown("---")

# High-Risk Area Recommendations
st.header("❗High-Risk Area Recommendations")
st.markdown("""
<div style="
    border: 2px solid #ff4b4b;
    background-color: #331111;
    padding: 1rem;
    border-radius: 10px;
    color: #ffffff;
">
<b>Living in or near a high-risk area?</b> Here are more drastic recommendations to reduce possibility of burglary if your area is in the top risk bracket as identified by our model.
</div>
""", unsafe_allow_html=True)
st.markdown("- Install **reinforced strike plates on doors**: provides more robust engagement with the door's lock.")
st.markdown("- Consider placing **internal reinforcement bars** or **window security film** to enhance security of windows.")
st.markdown("- Get a **dog**. They're one of the best deterrents for burglars.")
st.markdown("- Add **gravel** or **noisy path materials** around your house to make stealthy approaches more unlikely.")
st.markdown("Of course, these are substantial tips. Alongside or in place of these tactics, we recommend you to apply as many of the recommendations below as you can.")


# Individual-Level Recommendations
st.header("🏠 Individual-Level Recommendations")

st.success("🔒 Reinforce Doors and Windows")
st.markdown("- Use solid doors with deadbolts and robust window locks.")
st.markdown("- Always double-lock doors and secure all windows when leaving home (even for short trips).")

st.success("📹 Install Alarms & Cameras (Wisely)")
st.markdown("- Make them **visible** and use as part of a bigger security plan.")
st.markdown("- If you install an alarm, use it consistently and let it be known with signs/stickers.")

st.success("💡 WIDE Strategy")
st.markdown("- Apply **WIDE**: Window locks, Internal lights on timers, Door locks, External lights.")
st.markdown("- Use timer switches to make the house look occupied.")
st.markdown("- Install external motion lights to increase chances of burglar being seen.")

st.success("🏠 Make Your House Look Occupied")
st.markdown("- Leave a radio or TV on.")
st.markdown("- Ask neighbors to occasionally park in your driveway.")
st.markdown("- Avoid obvious signs of an empty house (like piled-up mail or bins left out).")

st.success("🎁 Protect Valuables")
st.markdown("- Hide valuables from view.")
st.markdown("- Use safes and mark items with UV pens or engravings linked to your address.")

st.success("🚪 Be Cautious with Uninvited Visitors")
st.markdown("- Verify identities before opening the door.")
st.markdown("- Especially important for vulnerable or elderly individuals.")
st.markdown("[For more information on Distraction Burglary](https://www.opendoorhomes.org/wp-content/uploads/2024/02/met-police-crime-prevention.pdf#:~:text=Overgrown%20bushes%20and%20trees%20make,to%20return%20to%20homes%20that)")

st.success("📵 Avoid Advertising Absence")
st.markdown("- Don’t post holiday updates in real time.")
st.markdown("- Avoid leaving notes that suggest you're away.")
st.markdown("[Article on social media's effect on burglary](https://www.independent.co.uk/travel/news-and-advice/holidays-empty-homes-burglaries-warning-instagram-facebook-tagging-a8967066.html#:~:text=Law%20enforcement%20agencies%20have%20warned,burglars%20to%20their%20empty%20properties)")

st.markdown("---")

# Community-Level Recommendations
st.header("🌍 Community-Level Recommendations")

st.success("🤝 Start or Join a **Neighbourhood Watch**")
st.markdown("- Strong community networks reduce crime by 16–26%.")
st.markdown("- Report suspicious activity and support your neighbors.")
st.markdown("- Look out for each other’s properties, share information, and collectively deter criminals.")

st.success("📣 Raise Awareness After Incidents")
st.markdown("- Warn others if a burglary occurs nearby.")
st.markdown("- Helps prevent 'near-repeat' crimes.")

st.success("🏘️ Improve Environmental Design (CPTED)")
st.markdown("- CPTED: Crime Prevention Through Environmental Design.")
st.markdown("- [Ensure good lighting on streets and paths](https://www.ojp.gov/ncjrs/virtual-library/abstracts/improved-street-lighting-and-crime-prevention-systematic-review#:~:text=Review%20www,Since)")
st.markdown("- Remove overgrown shrubs to reduce possible hiding places.")
st.markdown("- In shared buildings, secure rooftop or basement access point.")
st.markdown("- Encourage secure gates, fences, and access controls.")

