import streamlit as st
from auth import login
from pages.premium import premium_page
from pages.normal import normal_page

st.set_page_config(page_title="Survey RAG App", layout="wide")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None

# Render based on login state
if not st.session_state.logged_in:
    login()
else:
    # Sidebar with logout
    with st.sidebar:
        st.write(f"User type: {st.session_state.user_type}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    # Render pages based on user type
    if st.session_state.user_type == "premium":
        premium_page()
    else:
        normal_page()
