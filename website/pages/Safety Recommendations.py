import streamlit as st

st.title("Safety Recommendations")

# Individual-Level Recommendations
st.header("Individual-Level")
st.markdown("""
- **Reinforce Doors and Windows:** Use solid doors with deadbolts and robust window locks. Double-lock doors and latch all windows whenever you go out, even for short trips.
- **Target Harden with “WIDE”:** Apply the WIDE combo (Window locks, Internal lights on timers, Door locks, External lights) to deter break-ins. Use timer switches and external motion sensors.
- **Making The House Look Occupied:** Use timer lights, leave the radio or TV on, ask neighbors to park in your driveway occasionally, and avoid signs of an empty house.
- **Install Alarms and Cameras (Wisely):** Use visible alarms and cameras consistently and as part of a bigger security plan.
- **Protect Valuables (and Mark Them):** Keep valuables out of sight, use safes, and mark property with UV pens or engravings linked to your address.
- **Beware of Uninvited Visitors:** Verify identity of unexpected visitors and refuse entry if unsure, especially for elderly.
- **Don’t Advertise Your Absence:** Avoid posting real-time updates about being away on social media or leaving notes indicating absence.
""")

# Community-Level Recommendations
st.header("Community Level")
st.markdown("""
- **Start or Join a Neighborhood Watch:** Collaborate with neighbors to deter crime. Active neighborhood watches reduce crime by 16-26%.
- **Raise Community Awareness (especially after incidents):** Warn neighbors quickly after a burglary occurs to reduce “near repeats.”
- **Improve Environmental Design (CPTED):** Ensure streets are well-lit, remove hiding spots, secure access points, and advocate for crime-preventive designs in your community.
""")
