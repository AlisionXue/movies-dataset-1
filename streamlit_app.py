# import altair as alt
# import pandas as pd
# import streamlit as st
# import logging
# import pickle
# import requests

# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # -------------------------------------------------------------
# # 1. Logging Configuration
# # -------------------------------------------------------------
# logging.basicConfig(
#     filename="app_log.txt",    # Name of the log file
#     level=logging.INFO,        # Logging level
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logging.info("User accessed the system.")  # Log an access event

# # -------------------------------------------------------------
# # 2. Streamlit Page Setup
# # -------------------------------------------------------------
# st.set_page_config(
#     page_title="Movie Trends & Recommender",
#     page_icon="🎥"
# )

# # Initialize a session state variable
# if "user_session_active" not in st.session_state:
#     st.session_state["user_session_active"] = True

# # -------------------------------------------------------------
# # 3. Data Loading and Preparation
# # -------------------------------------------------------------
# @st.cache_data
# def load_movies_summary():
#     """
#     Loads the main CSV file that contains summary data about movies.
#     Returns a pandas DataFrame.
#     """
#     return pd.read_csv("data/movies_genres_summary.csv")

# df = load_movies_summary()

# # Load the movie dictionary and similarity matrix
# movie_dict = pickle.load(open("data/movies_dict.pkl", "rb"))
# movies_list = pd.DataFrame(movie_dict)

# similarity_data = pickle.load(open("data/similarity.pkl", "rb"))

# # -------------------------------------------------------------
# # 4. Helper Functions
# # -------------------------------------------------------------
# def fetch_poster(movie_id):
#     """
#     Fetches the poster URL for the given movie ID using the TMDB API.
#     """
#     api_key = '1841b88ac1115b2ca3334950056976c2'  # Same API key
#     api_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
#     response = requests.get(api_url)

#     if response.status_code == 200:
#         movie_data = response.json()
#         poster_path = movie_data.get("poster_path")
#         if poster_path:
#             return f"https://image.tmdb.org/t/p/w500{poster_path}"
#     return None

# def get_movie_recommendations(chosen_movie):
#     """
#     Given a movie title, returns a list of up to 5 recommended movies
#     along with their poster URLs, using a precomputed similarity matrix.
#     """
#     match_index = movies_list[movies_list["title"] == chosen_movie].index
#     if len(match_index) == 0:
#         return []

#     idx = match_index[0]
#     sims = similarity_data[idx]
#     ranking = sorted(
#         enumerate(sims),
#         key=lambda x: x[1],
#         reverse=True
#     )[1:6]  # Skip the first one (itself), then take next 5

#     recommendations = []
#     for movie_idx, score in ranking:
#         recommendations.append({
#             "title": movies_list.iloc[movie_idx]["title"],
#             "poster_url": fetch_poster(movies_list.iloc[movie_idx]["movie_id"])
#         })
#     return recommendations

# # -------------------------------------------------------------
# # 5. Main Application Flow
# # -------------------------------------------------------------
# if st.session_state["user_session_active"]:
#     st.title("🎥 Explore Movie Trends and Recommendations")
#     st.write(
#         """
#         Welcome! Use this application to analyze movie trends and discover 
#         films similar to your favorites. Choose a movie below to get suggestions.
#         """
#     )

#     # Movie selection widget
#     chosen_movie_title = st.selectbox(
#         "Select a movie for recommendations:",
#         movies_list["title"]
#     )

#     if st.button("Get Recommendations"):
#         movie_suggestions = get_movie_recommendations(chosen_movie_title)
#         if movie_suggestions:
#             st.subheader("Recommended Movies:")
#             rec_columns = st.columns(3)
#             for i, suggestion in enumerate(movie_suggestions):
#                 with rec_columns[i % 3]:
#                     st.write(suggestion["title"])
#                     if suggestion["poster_url"]:
#                         st.image(suggestion["poster_url"], use_column_width=True)
#         else:
#             st.error("No recommendations available for this title.")

#     # -------------------------------------------------------------
#     # 6. Simple Machine Learning Model (Movie Ratings Prediction)
#     # -------------------------------------------------------------
#     st.write("### Movie Ratings Prediction Example")

#     # Use only year and vote_average for a simple demonstration
#     df_ml = df[["year", "vote_average"]].dropna()
#     X = df_ml[["year"]]
#     y = df_ml["vote_average"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     lin_reg = LinearRegression()
#     lin_reg.fit(X_train, y_train)

#     y_pred = lin_reg.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)

#     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     st.write("**Actual vs. Predicted Ratings**")
#     results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#     st.dataframe(results_df)

#     # -------------------------------------------------------------
#     # 7. Basic Data Visualizations
#     # -------------------------------------------------------------
#     # Yearly movie counts (bar chart)
#     st.bar_chart(df.groupby("year").size())

#     # Genre distribution (pie chart)
#     genre_counts = df.groupby("genre").size()
#     fig, ax = plt.subplots(figsize=(8, 5))
#     genre_counts.plot.pie(
#         y="genre",
#         autopct="%.2f%%",
#         ax=ax,
#         title="Genre Distribution"
#     )
#     st.pyplot(fig)

#     # -------------------------------------------------------------
#     # 8. Filtering and Trend Analysis
#     # -------------------------------------------------------------
#     selected_genres = st.multiselect(
#         "Select genres:",
#         df["genre"].unique(),
#         default=["Action", "Comedy"]
#     )
#     year_min, year_max = st.slider(
#         "Select year range:",
#         min_value=1980,
#         max_value=2020,
#         value=(2000, 2015)
#     )

#     filtered_df = df[
#         (df["genre"].isin(selected_genres)) &
#         (df["year"].between(year_min, year_max))
#     ]

#     pivot_table = filtered_df.pivot_table(
#         index="year",
#         columns="genre",
#         values="gross",
#         aggfunc="sum",
#         fill_value=0
#     )
#     st.dataframe(pivot_table)

#     # Altair line chart
#     alt_data = pd.melt(
#         pivot_table.reset_index(),
#         id_vars="year",
#         var_name="genre",
#         value_name="gross"
#     )
#     alt_chart = (
#         alt.Chart(alt_data)
#         .mark_line()
#         .encode(
#             x="year:O",
#             y="gross:Q",
#             color="genre:N",
#             tooltip=["year", "genre", "gross"]
#         )
#         .properties(width=600, height=400)
#     )
#     st.altair_chart(alt_chart, use_container_width=True)

# else:
#     st.warning("Session is not active. Please log in or verify your credentials.")


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

# Initialize a session state variable
if "user_session_active" not in st.session_state:
    st.session_state["user_session_active"] = False

# -------------------------------------------------------------
# 3. Data Loading and Preparation
# -------------------------------------------------------------
@st.cache_data
def load_movies_summary():
    """
    Loads the main CSV file that contains summary data about movies.
    Returns a pandas DataFrame.
    """
    return pd.read_csv("data/movies_genres_summary.csv")

df = load_movies_summary()

# Load the movie dictionary and similarity matrix
movie_dict = pickle.load(open("data/movies_dict.pkl", "rb"))
movies_list = pd.DataFrame(movie_dict)

similarity_data = pickle.load(open("data/similarity.pkl", "rb"))

# -------------------------------------------------------------
# 4. Helper Functions
# -------------------------------------------------------------
def fetch_poster(movie_id):
    """
    Fetches the poster URL for the given movie ID using the TMDB API.
    """
    api_key = '1841b88ac1115b2ca3334950056976c2'  # Same API key
    api_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(api_url)

    if response.status_code == 200:
        movie_data = response.json()
        poster_path = movie_data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def get_movie_recommendations(chosen_movie):
    """
    Given a movie title, returns a list of up to 5 recommended movies
    along with their poster URLs, using a precomputed similarity matrix.
    """
    match_index = movies_list[movies_list["title"] == chosen_movie].index
    if len(match_index) == 0:
        return []

    idx = match_index[0]
    sims = similarity_data[idx]
    ranking = sorted(
        enumerate(sims),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # Skip the first one (itself), then take next 5

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
    st.title("🎥 Explore Movie Trends and Recommendations")
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

    # Use only year and vote_average for a simple demonstration
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
    # Yearly movie counts (bar chart)
    st.bar_chart(df.groupby("year").size())

    # Genre distribution (pie chart)
    genre_counts = df.groupby("genre").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_counts.plot.pie(
        y="genre",
        autopct="%.2f%%",
        ax=ax,
        title="Genre Distribution"
    )
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
        (df["genre"].isin(selected_genres)) &
        (df["year"].between(year_min, year_max))
    ]

    pivot_table = filtered_df.pivot_table(
        index="year",
        columns="genre",
        values="gross",
        aggfunc="sum",
        fill_value=0
    )
    st.dataframe(pivot_table)

    # Altair line chart
    alt_data = pd.melt(
        pivot_table.reset_index(),
        id_vars="year",
        var_name="genre",
        value_name="gross"
    )
    alt_chart = (
        alt.Chart(alt_data)
        .mark_line()
        .encode(
            x="year:O",
            y="gross:Q",
            color="genre:N",
            tooltip=["year", "genre", "gross"]
        )
        .properties(width=600, height=400)
    )
    st.altair_chart(alt_chart, use_container_width=True)

else:
    with st.form("login_form"):
        username = st.text_input("username")
        password = st.text_input("password", type="password")
        submitted = st.form_submit_button("login test")

        if submitted:
            if username == "admin" and password == "123":
                st.session_state["user_session_active"] = True
                st.rerun()
            else:
                st.error("Error")
