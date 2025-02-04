import streamlit as st

# 预设的用户名和密码
USERNAME = "admin"
PASSWORD = "123"

# 检查登录状态
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# 如果已登录
if st.session_state["user_session_active"]:
    st.success(f"Welcome, {USERNAME}!")
    st.write("You have successfully accessed the secured application.")
    if st.button("Logout"):
        st.session_state["user_session_active"] = False
        st.experimental_rerun()

# 如果未登录
else:
    st.warning("Please login to access the application.")
    with st.form("login_form"):
        user_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if user_input == USERNAME and password_input == PASSWORD:
                st.success("Login successful!")
                st.session_state["user_session_active"] = True
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password.")
