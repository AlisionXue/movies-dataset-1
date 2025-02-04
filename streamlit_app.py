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
    filename="app_log.txt",    # Log file name
    level=logging.INFO,        # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("User accessed the system.")

# -------------------------------------------------------------
# 2. Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Movie Trends & Recommender",
    page_icon="🎥"
)

# Initialize session state variable if not already set
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# Always get the latest query parameters
query_params = st.experimental_get_query_params()
url_username = query_params.get("username", [None])[0]
url_password = query_params.get("password", [None])[0]

# Log only the username for security reasons
logging.info(f"Query parameters received - username: {url_username}")

# Predefined username and password
USERNAME = "admin"
PASSWORD = "123"

# Automatic login based on URL parameters
# Note: Auto-login works reliably on a fresh load.
if url_username == USERNAME and url_password == PASSWORD:
    st.session_state["user_session_active"] = True
    logging.info("Automatic login successful.")
else:
    logging.info("Automatic login failed. Session state not activated.")

# -------------------------------------------------------------
# 3. Data Loading and Preparation
# -------------------------------------------------------------
@st.cache_data
def load_movies_summary():
    """Loads the main CSV file containing movie summary data."""
    return pd.read_csv("data/movies_genres_summary.csv")

df = load_movies_summary()

@st.cache_data
def load_pickle_data():
    movie_dict = pickle.load(open("data/movies_dict.pkl", "rb"))
    similarity_data = pickle.load(open("data/similarity.pkl", "rb"))
    movies_list = pd.DataFrame(movie_dict)
    return movies_list, similarity_data

movies_list, similarity_data = load_pickle_data()

# -------------------------------------------------------------
# 4. Helper Functions
# -------------------------------------------------------------
def fetch_poster(movie_id):
    """Fetches the poster URL for the given movie ID using the TMDB API."""
    api_key = '1841b88ac1115b2ca3334950056976c2'  # TMDB API key
    api_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        movie_data = response.json()
        poster_path = movie_data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException as e:
        logging.error(f"Error fetching poster for movie_id {movie_id}: {e}")
    return None

def get_movie_recommendations(chosen_movie):
    """
    Given a movie title, returns a list of up to 5 recommended movies
    along with their poster URLs using a precomputed similarity matrix.
    """
    match_index = movies_list[movies_list["title"] == chosen_movie].index
    if len(match_index) == 0:
        return []
    
    idx = match_index[0]
    sims = similarity_data[idx]
    ranking = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for movie_idx, score in ranking:
        recommendations.append({
            "title": movies_list.iloc[movie_idx]["title"],
            "poster_url": fetch_poster(movies_list.iloc[movie_idx]["movie_id"])
        })
    return recommendations

# -------------------------------------------------------------
# 5. Main Application Flow
# -------------------------------------------------------------
if st.session_state["user_session_active"]:
    st.title("🎥 Movie Trends and Recommendations System")
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
                        st.image(suggestion["poster_url"], use_container_width=True)
        else:
            st.error("No recommendations available for this title.")

    # -------------------------------------------------------------
    # 6. Simple Machine Learning Model (Movie Ratings Prediction)
    # -------------------------------------------------------------
    st.write("### Movie Ratings Prediction Example")
    df_ml = df[["year", "vote_average"]].dropna()
    X = df_ml[["year"]]
    y = df_ml["vote_average"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write("**Actual vs. Predicted Ratings**")
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results_df)

    # -------------------------------------------------------------
    # 7. Basic Data Visualizations
    # -------------------------------------------------------------
    st.bar_chart(df.groupby("year").size())
    genre_counts = df.groupby("genre").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_counts.plot.pie(y="genre", autopct="%.2f%%", ax=ax, title="Genre Distribution")
    st.pyplot(fig)

    # -------------------------------------------------------------
    # 8. Filtering and Trend Analysis
    # -------------------------------------------------------------
    selected_genres = st.multiselect(
        "Select genres:",
        df["genre"].unique(),
        default=["Action", "Comedy"]
    )
    year_min, year_max = st.slider(
        "Select year range:",
        min_value=1980,
        max_value=2020,
        value=(2000, 2015)
    )
    filtered_df = df[
        (df["genre"].isin(selected_genres)) & (df["year"].between(year_min, year_max))
    ]
    pivot_table = filtered_df.pivot_table(
        index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
    )
    st.dataframe(pivot_table)
    
    alt_data = pd.melt(
        pivot_table.reset_index(), id_vars="year", var_name="genre", value_name="gross"
    )
    alt_chart = (
        alt.Chart(alt_data)
        .mark_line()
        .encode(
            x="year:O", y="gross:Q", color="genre:N", tooltip=["year", "genre", "gross"]
        )
        .properties(width=600, height=400)
    )
    st.altair_chart(alt_chart, use_container_width=True)

    if st.button("Logout"):
        st.session_state["user_session_active"] = False
        st.experimental_rerun()

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
