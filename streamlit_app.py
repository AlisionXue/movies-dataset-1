import altair as alt
import pandas as pd
import streamlit as st
import logging
import pickle
import requests

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------
# 1. Logging Configuration
# -------------------------------------------------------------
logging.basicConfig(
    filename="app_log.txt",    # Name of the log file
    level=logging.INFO,        # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("User accessed the system.")  # Log an access event

# -------------------------------------------------------------
# 2. Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Movie Trends & Recommender",
    page_icon="🎥"
)

# Extract query parameters using st.query_params (NEW)
query_params = st.query_params
pre_filled_username = query_params.get("username", "")
pre_filled_password = query_params.get("password", "")

# Initialize a session state variable
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# -------------------------------------------------------------
# 3. Login Handling
# -------------------------------------------------------------
if not st.session_state["user_session_active"]:
    with st.form("login_form"):
        username = st.text_input("Username", value=pre_filled_username)
        password = st.text_input("Password", value=pre_filled_password, type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username == "admin" and password == "123":
                st.session_state["user_session_active"] = True
                st.success("Login successful! Redirecting...")
                st.rerun()  # Fixed rerun issue (Replaces experimental_rerun)
            else:
                st.error("Incorrect username or password.")

# -------------------------------------------------------------
# 4. Main Application Flow (Only if logged in)
# -------------------------------------------------------------
if st.session_state["user_session_active"]:
    st.title("🎥 Movie Trends & Recommender System")
    st.write(
        """
        Welcome! Use this application to analyze movie trends and discover 
        films similar to your favorites. Choose a movie below to get suggestions.
        """
    )

    # Movie selection widget
    chosen_movie_title = st.selectbox(
        "Select a movie for recommendations:",
        movies_list["title"]
    )

    if st.button("Get Recommendations"):
        movie_suggestions = get_movie_recommendations(chosen_movie_title)
        if movie_suggestions:
            st.subheader("Recommended Movies:")
            rec_columns = st.columns(3)
            for i, suggestion in enumerate(movie_suggestions):
                with rec_columns[i % 3]:
                    st.write(suggestion["title"])
                    if suggestion["poster_url"]:
                        st.image(suggestion["poster_url"], use_column_width=True)
        else:
            st.error("No recommendations available for this title.")
