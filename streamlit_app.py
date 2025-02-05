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
    filename="app_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("User accessed the system.")

# -------------------------------------------------------------
# 2. Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Movie Trends & Recommender System",
    page_icon="🎥"
)

# Initialize session state
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# -------------------------------------------------------------
# 3. Data Loading and Preparation
# -------------------------------------------------------------
@st.cache_data
def load_movies_summary():
    return pd.read_csv("data/movies_genres_summary.csv")

df = load_movies_summary()

movie_dict = pickle.load(open("data/movies_dict.pkl", "rb"))
movies_list = pd.DataFrame(movie_dict)

similarity_data = pickle.load(open("data/similarity.pkl", "rb"))

# -------------------------------------------------------------
# 4. Helper Functions
# -------------------------------------------------------------
def fetch_poster(movie_id):
    api_key = '1841b88ac1115b2ca3334950056976c2'
    api_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(api_url)
    if response.status_code == 200:
        movie_data = response.json()
        poster_path = movie_data.get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

# Ensure function is defined before use
def get_movie_recommendations(chosen_movie):
    if "get_movie_recommendations" not in globals():
        st.error("Function get_movie_recommendations is not defined!")
        return []
    
    match_index = movies_list[movies_list["title"] == chosen_movie].index
    if len(match_index) == 0:
        return []
    
    idx = match_index[0]
    sims = similarity_data[idx]
    ranking = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = []
    for movie_idx, _ in ranking:
        recommendations.append({
            "title": movies_list.iloc[movie_idx]["title"],
            "poster_url": fetch_poster(movies_list.iloc[movie_idx]["movie_id"])
        })
    return recommendations

# -------------------------------------------------------------
# 5. Main Application Flow
# -------------------------------------------------------------
if st.session_state["user_session_active"]:
    st.title("🎥 Movie Trends & Recommender System")
    st.write("Select a movie to get recommendations.")
    chosen_movie_title = st.selectbox("Select a movie:", movies_list["title"])
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
            st.error("No recommendations available.")
else:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "123":
                st.session_state["user_session_active"] = True
                st.rerun()
            else:
                st.error("Invalid login credentials.")
