import streamlit as st

# Dummy user database (replace with DB or API later)
USERS = {
    "user1": {"password": "123", "type": "normal"},
    "user2": {"password": "abc", "type": "premium"},
}

def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["user_type"] = USERS[username]["type"]
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
