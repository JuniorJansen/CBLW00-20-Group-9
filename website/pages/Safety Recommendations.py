import streamlit as st


# MUST be the first Streamlit command
st.set_page_config(page_title="Safety Recommendations", layout="centered")

# Background styling
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

st.title("🛡️ Safety Recommendations")
st.markdown("Protect yourself and your community with these strategies.")
st.markdown("[See the European Crime Prevention Guide for Preventing Domestic Burglaries](https://eucpn.org/sites/default/files/document/files/2106_The%20prevention%20of%20domestic%20burglary_LR.pdf#:~:text=night%20owing%20to%20the%20increased,9)")
st.markdown("---")

# Individual-Level Recommendations
st.header("🏠 Individual-Level Recommendations")

st.success("🔒 Reinforce Doors and Windows")
st.markdown("- Use solid doors with deadbolts and robust window locks.")
st.markdown("- Always double-lock doors and secure all windows when leaving home.")

st.success("💡 WIDE Strategy")
st.markdown("- Apply **WIDE**: Window locks, Internal lights on timers, Door locks, External lights.")
st.markdown("- Use timer switches and external motion lights.")

st.success("🏠 Make Your House Look Occupied")
st.markdown("- Use timer lights, leave a radio or TV on.")
st.markdown("- Ask neighbors to park in your driveway.")

st.success("📹 Install Alarms & Cameras (Wisely)")
st.markdown("- Make them **visible** and use as part of a full security plan.")

st.success("🎁 Protect Valuables")
st.markdown("- Hide valuables from view.")
st.markdown("- Use safes and mark items with UV pens or engravings linked to your address.")

st.success("🚪 Be Cautious with Visitors")
st.markdown("- Verify identities before opening the door.")
st.markdown("- Especially important for vulnerable or elderly individuals.")

st.success("📵 Avoid Advertising Absence")
st.markdown("- Don’t post holiday updates in real time.")
st.markdown("- Avoid leaving notes that suggest you're away.")

st.markdown("---")

# Community-Level Recommendations
st.header("🌍 Community-Level Recommendations")

st.success("🤝 Start or Join a **Neighbourhood Watch**")
st.markdown("- Strong community networks reduce crime by **16–26%**.")
st.markdown("- Report suspicious activity and support your neighbors.")

st.success("📣 Raise Awareness After Incidents")
st.markdown("- Warn others if a burglary occurs nearby.")
st.markdown("- Helps prevent 'near-repeat' crimes.")

st.success("🏘️ Improve Environmental Design (CPTED)")
st.markdown("- Ensure good lighting on streets and paths.")
st.markdown("- Remove overgrown shrubs and hiding places.")
st.markdown("- Encourage secure gates, fences, and access controls.")

st.markdown("---")
st.caption("These recommendations are informed by criminological research and designed to reduce residential burglary risk in London.")

