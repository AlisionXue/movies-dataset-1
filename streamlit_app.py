import altair as alt
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import logging
import pickle
import requests
import gdown
import os


# 配置日志记录
logging.basicConfig(
    filename="app.log",  # 日志文件
    level=logging.INFO,  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)

# 示例日志记录
logging.info("System detection user access ")


# Show the page title and description.
st.set_page_config(page_title="Movie data display and recommend system", page_icon="🎬")



if 'init_flag' not in st.session_state:
    st.session_state.init_flag = True
    st.session_state.login_flag = False
    
    # 定义文件路径
    file_path = "data/similarity.pkl"
    
    # 如果文件不存在，则下载
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id=1_UeAu9mJdJD0Hqt9YoFaxj7dgb-cupFy"
        gdown.download(url, file_path, quiet=False)


if st.session_state.login_flag : 
    st.title("🎬 Movie data diaplay and recommend system")
    st.write(
        """
        This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
        It shows which movie genre performed best at the box office over the years. Just 
        click on the widgets below to explore!
        """
    )


    # Load the data from a CSV. We're caching this so it doesn't reload every time the app
    # reruns (e.g. if the user interacts with the widgets).
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/movies_genres_summary.csv")
        return df


    df = load_data()
    
        
    # Load data
    movies_dict = pickle.load(open('data/movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)

    similarity = pickle.load(open('data/similarity.pkl', 'rb'))
        
    # Function to fetch movie poster from TMDb API
    def fetch_poster(movie_id):
        API_KEY = '1841b88ac1115b2ca3334950056976c2'
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        return None
    # Function to recommend movies
    def recommended(movie):
        movie_index = movies[movies['title'] == movie].index
        if len(movie_index) == 0:
            return []  # Return empty list if movie not found
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_movies = []
        
        for i in movie_list:
            recommended_movies.append({
                'title': movies.iloc[i[0]]['title'],
                'poster': fetch_poster(movies.iloc[i[0]]['movie_id'])  # Fetch poster for each recommended movie
            })
        return recommended_movies
 
    # Title and sidebar
    selected_movie_name = st.selectbox('Select a movie:', movies['title'])
    recommend_button = st.button('Recommend')

    # Main content
    if recommend_button:
        recommendations = recommended(selected_movie_name)
        if recommendations:
            st.subheader("Recommended Movies")
            # Display posters in a grid layout
            col1, col2, col3 = st.columns(3)
            for movie in recommendations:
                with col1, st.expander(movie['title']):
                    if movie['poster']:
                        st.image(movie['poster'], caption='', use_container_width=True)
                    else:
                        st.write("No poster available")
        else:
            st.error("No recommendations found.")










    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error


    # 添加机器学习部分
    st.write("### Machine Learning: Predict Ratings")

    # 选择需要的列
    df_ml = df[['year', 'vote_average']].dropna()
    X = df_ml[['year']]
    y = df_ml['vote_average']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测并评估
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    # 显示结果
    st.write(f"#### Model Mean Squared Error: {mse:.2f}")
    st.write("#### Predictions")
    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))





    data_by_year = df.groupby('year').size()
    # st.write(data_by_year)
    # a = data_by_year.plot.bar(title='Number of movies per year', figsize=(20,10))
    st.bar_chart(data_by_year)

    st.bar_chart(df['vote_average'])
    data_by_genre = df.groupby('genre').size()
    fig, ax = plt.subplots(figsize=(10, 6))  # 设置图表大小
    data_by_genre.plot.pie( title='Number of movies per genre') 

    # 将饼状图嵌入到 Streamlit
    st.pyplot(fig)



    # Show a multiselect widget with the genres using `st.multiselect`.
    genres = st.multiselect(
        "Genres",
        df.genre.unique(),
        ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
    )

    # Show a slider widget with the years using `st.slider`.
    years = st.slider("Years", 1986, 2006, (2000, 2016))

    # Filter the dataframe based on the widget input and reshape it.
    df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]
    df_reshaped = df_filtered.pivot_table(
        index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
    )
    df_reshaped = df_reshaped.sort_values(by="year", ascending=False)


    # Display the data as a table using `st.dataframe`.
    st.dataframe(
        df_reshaped,
        use_container_width=True,
        column_config={"year": st.column_config.TextColumn("Year")},
    )

    # Display the data as an Altair chart using `st.altair_chart`.
    df_chart = pd.melt(
        df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
    )
    chart = (
        alt.Chart(df_chart)
        .mark_line()
        .encode(
            x=alt.X("year:N", title="Year"),
            y=alt.Y("gross:Q", title="Gross earnings ($)"),
            color="genre:N",
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    username = st.text_input(label='username')
    password  = st.text_input(label='password')
    if st.button('login'):
        if password == '123':
            st.session_state.login_flag= True
            st.rerun()
