import streamlit as st

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Police Insights", layout="centered")

# 🔧 Background styling
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

# Password that police has access to
PASSWORD = "police123"

# Initialize session state for authentication
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

# If not authenticated, show login form
if not st.session_state.auth_ok:
    st.subheader("🔒 Enter password to access Police Insights")
    with st.form("password_form"):
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Submit")

        if submit:
            if password == PASSWORD:
                st.session_state.auth_ok = True
                st.success("✅ Access granted! You may now view police insights.")
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    st.stop()
else:
    st.title("🚓 Police Insights")
    st.write("Welcome! Here are your restricted insights:")
    # Add your sensitive data, graphs, or analysis here
