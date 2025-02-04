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

# Extract query parameters correctly
pre_filled_username = st.query_params.get("username", [""])[0]
pre_filled_password = st.query_params.get("password", [""])[0]

# Initialize a session state variable
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# -------------------------------------------------------------
# 3. Data Loading (Ensure movies_list is loaded before use)
# -------------------------------------------------------------
@st.cache_data
def load_movie_data():
    movie_dict = pickle.load(open("data/movies_dict.pkl", "rb"))
    return pd.DataFrame(movie_dict)

movies_list = load_movie_data()  # Load movie data

# -------------------------------------------------------------
# 4. Login Handling
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
                st.rerun()  # Corrected rerun function
            else:
                st.error("Incorrect username or password.")

# -------------------------------------------------------------
# 5. Main Application Flow (Only if logged in)
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
        movies_list["title"] if not movies_list.empty else ["No movies available"]
    )

    if st.button("Get Recommendations") and not movies_list.empty:
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
