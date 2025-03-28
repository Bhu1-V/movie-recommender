import pandas as pd
import numpy as np
import streamlit as st
import scipy.sparse
import joblib
import json

# --- Load Poster Paths with Error Handling ---
try:
    with open('posters.json', 'r') as f:
        poster_paths = json.load(f)
except FileNotFoundError:
    poster_paths = {}
    st.warning("**Warning**: posters.json not found. Posters will not be displayed.")
except json.JSONDecodeError:
    poster_paths = {}
    st.warning("**Warning**: posters.json is corrupted. Posters will not be displayed.")

# --- Data Loading and Preprocessing ---
@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess movie data from CSV files with reduced memory usage.
    
    Always loads a random 10% subset of the ratings.
    """
    try:
        # Load only necessary columns with efficient dtypes
        ratings = pd.read_csv(
            'ml-latest-small/ratings.csv',
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}
        )
        movies = pd.read_csv(
            'ml-latest-small/movies.csv',
            dtype={'movieId': 'int32'}
        )
        links = pd.read_csv(
            'ml-latest-small/links.csv',
            usecols=['movieId', 'imdbId'],
            dtype={'movieId': 'int32', 'imdbId': 'int32'}
        )
    except FileNotFoundError as e:
        st.error(f"**Error**: {e}. Please ensure CSV files are in 'ml-latest-small'.")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"**Error**: Failed to parse CSV files. {e}")
        st.stop()

    # Always sample 10% of the ratings
    ratings = ratings.sample(frac=0.1, random_state=42).reset_index(drop=True)
    st.info(f"Using a 10% sample of the ratings dataset ({len(ratings)} ratings).")

    # Handle missing values
    if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
        st.warning("**Warning**: Missing values in ratings. Dropping affected rows.")
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])

    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    rating_dist = ratings.groupby('movieId')['rating'].value_counts(normalize=True).unstack(fill_value=0)
    return ratings, movies, links, movie_stats, rating_dist

try:
    ratings, movies, links, movie_stats, rating_dist = load_data()
except Exception as e:
    st.error(f"**Error**: Failed to load data. {e}")
    st.stop()

# --- Load Precomputed Similarity Matrix ---
@st.cache_data(show_spinner=False)
def load_similarity_matrix():
    try:
        sim_matrix_sparse = scipy.sparse.load_npz('similarity_matrix.npz')
        return sim_matrix_sparse.toarray()
    except FileNotFoundError:
        st.error("**Error**: similarity_matrix.npz not found. Please upload it.")
        return np.array([])

similarity_matrix = load_similarity_matrix()

# --- Load Precomputed SVD Model ---
@st.cache_resource(show_spinner=False)
def load_svd_model():
    try:
        model = joblib.load('svd_model.joblib')
        trainset = joblib.load('trainset.joblib')
        return model, trainset
    except FileNotFoundError:
        st.error("**Error**: SVD model or trainset files not found. Please upload them.")
        return None, None

model, trainset = load_svd_model()

# --- Popularity-Based Recommender ---
@st.cache_data(show_spinner=False)
def compute_popularity_based_recommendations(ratings, movies):
    """Compute top 10 movies based on weighted ratings."""
    C = ratings['rating'].mean()
    num_ratings = ratings.groupby('movieId').size()
    m = num_ratings.quantile(0.9)
    movie_stats_local = ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
    movie_stats_local.columns = ['num_ratings', 'avg_rating']
    qualified_movies = movie_stats_local[movie_stats_local['num_ratings'] >= m].copy()
    qualified_movies['weighted_rating'] = (
        (qualified_movies['num_ratings'] / (qualified_movies['num_ratings'] + m)) * qualified_movies['avg_rating'] +
        (m / (qualified_movies['num_ratings'] + m)) * C
    )
    top_movies = qualified_movies.sort_values('weighted_rating', ascending=False).head(10)
    top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId', how='left')
    return top_movies

try:
    top_movies = compute_popularity_based_recommendations(ratings, movies)
except Exception as e:
    st.error(f"**Error**: Failed to compute popularity-based recommendations. {e}")
    top_movies = pd.DataFrame()


# --- Recommendation Functions ---
def get_content_based_recommendations(selected_titles, similarity_matrix, movies, k=10):
    """Get content-based recommendations based on genre similarity."""
    selected_indices = [movies[movies['title'] == title].index[0] for title in selected_titles if title in movies['title'].values]
    if not selected_indices:
        st.warning("**Warning**: Selected movies not found in dataset.")
        return []
    avg_similarity = np.mean(similarity_matrix[selected_indices], axis=0)
    avg_similarity[selected_indices] = 0
    top_indices = np.argsort(avg_similarity)[::-1][:k]
    insights = []
    for idx in top_indices:
        title = movies.iloc[idx]['title']
        genres_str = movies.iloc[idx]['genres']
        genres_list = genres_str.split('|')
        selected_genres = set.union(*[set(movies.iloc[i]['genres'].split('|')) for i in selected_indices])
        overlapping_genres = set(genres_list).intersection(selected_genres)
        similarity_score = avg_similarity[idx]
        top_genres = list(overlapping_genres)[:3]
        insight = {
            'overlapping_genres': list(overlapping_genres),
            'similarity_score': float(similarity_score),
            'top_genres': top_genres
        }
        insights.append((title, insight))
    return insights

# --- SVD Recommendation Function ---
def get_recommendations(selected_ids, model, trainset, k=10):
    """Get recommendations using SVD by simulating a pseudo-user who likes the selected movies."""
    if model is None or trainset is None:
        st.error("**Error**: SVD model unavailable.")
        return {}

    # Create a pseudo-user ID (one greater than the max user ID in the trainset)
    pseudo_user_id = max(trainset.all_users()) + 1

    # Get all items (movies) in the trainset
    all_items = set(trainset.all_items())

    # Exclude the selected movies from the candidates
    candidate_items = [item for item in all_items if trainset.to_raw_iid(item) not in selected_ids]

    # Predict ratings for all candidate items for the pseudo-user
    predictions = [model.predict(pseudo_user_id, trainset.to_raw_iid(item), r_ui=5.0 if trainset.to_raw_iid(item) in selected_ids else None)
                   for item in candidate_items]

    # Sort predictions by estimated rating and take top k
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

    recommended = {}
    for pred in top_predictions:
        raw_id = pred.iid
        recommended[raw_id] = {
            'predicted_rating': pred.est,
            'similar_to': ', '.join([id_to_title[mid] for mid in selected_ids if mid in id_to_title])
        }
    return recommended

# Mappings between movie titles and IDs
title_to_id = dict(zip(movies['title'], movies['movieId']))
id_to_title = dict(zip(movies['movieId'], movies['title']))

# --- Streamlit Web App ---
st.title("Movie Recommender System")
st.markdown("""
Welcome to the Movie Recommender System! Choose a recommendation type from the sidebar:
- **Popularity-based**: Top 10 movies based on weighted ratings.
- **Collaborative Filtering (SVD)**: Personalized recommendations based on user rating patterns.
- **Content-Based**: Recommendations based on genre similarities.

**Note**: To optimize performance, this app uses a 10% sample of the MovieLens ratings dataset.
""")

recommendation_type = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["Popularity-based", "Collaborative Filtering (SVD)", "Content-Based"]
)

if recommendation_type == "Popularity-based":
    st.subheader("Top 10 Popular Movies")
    st.markdown("""
    Ranked using: `(v/(v+m) * R) + (m/(v+m) * C)`  
    where `v` = number of ratings, `R` = average rating, `m` = 90th percentile of ratings, `C` = mean rating.
    """)
    if top_movies.empty:
        st.error("**Error**: No popularity-based recommendations available.")
    else:
        for idx, row in top_movies.iterrows():
            movie_id = row['movieId']
            movie_id_str = f"{movie_id}.0"
            col1, col2 = st.columns([1, 3])
            with col1:
                if movie_id_str in poster_paths:
                    try:
                        st.image(poster_paths[movie_id_str], width=100)
                    except Exception:
                        st.write("Poster unavailable")
                else:
                    st.write("No poster")
            with col2:
                st.write(f"**{row['title']}** - Weighted Rating: {row['weighted_rating']:.2f}")

elif recommendation_type == "Collaborative Filtering (SVD)":
    st.subheader("Personalized Recommendations (SVD)")
    st.markdown("""
    Based on user rating patterns. Insights include predicted ratings for a pseudo-user who likes the selected movies.
    """)
    movie_list = movies['title'].tolist()
    selected_movies = st.multiselect("Select Movies You Like", movie_list)
    if selected_movies:
        selected_ids = [title_to_id[title] for title in selected_movies if title in title_to_id]
        if not selected_ids:
            st.warning("**Warning**: Selected movies not found in dataset.")
        else:
            recommended_dict = get_recommendations(selected_ids, model, trainset, k=10)
            recommended_ids = list(recommended_dict.keys())
            recommended_titles = [id_to_title[mid] for mid in recommended_ids if mid in id_to_title]
            
            st.write("**Recommended Movies:**")
            if recommended_titles:
                for title in recommended_titles:
                    movie_id = title_to_id[title]
                    movie_id_str = f"{movie_id}.0"
                    imdb_url = "#"
                    if movie_id in links['movieId'].values:
                        imdb_id = links[links['movieId'] == movie_id]['imdbId'].values[0]
                        imdb_url = f"https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/"
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if movie_id_str in poster_paths:
                            try:
                                st.image(poster_paths[movie_id_str], width=100)
                            except Exception:
                                st.write("Poster unavailable")
                        else:
                            st.write("No poster")
                    with col2:
                        st.write(f"**{title}**")
                        if movie_id in recommended_dict:
                            predicted_rating = recommended_dict[movie_id]['predicted_rating']
                            similar_to = recommended_dict[movie_id]['similar_to']
                            avg_rating = movie_stats.loc[movie_id, 'avg_rating'] if movie_id in movie_stats.index else 'N/A'
                            num_ratings = movie_stats.loc[movie_id, 'num_ratings'] if movie_id in movie_stats.index else 'N/A'
                            genres_str = movies[movies['movieId'] == movie_id]['genres'].values[0]
                            rating_5_pct = rating_dist.loc[movie_id, 5.0] * 100 if movie_id in rating_dist.index and 5.0 in rating_dist.columns else 0
                            
                            st.write(f"- **Why Recommended**: Predicted rating {predicted_rating:.2f}/5 based on your liking for '{similar_to}'.")
                            st.write(f"- **Genres**: {genres_str.replace('|', ', ')}")
                            st.write(f"- **Rating**: Avg {avg_rating:.2f} ({num_ratings} ratings); {rating_5_pct:.1f}% 5 stars" if avg_rating != 'N/A' else "- **Rating**: N/A")
                            st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
            else:
                st.write("No recommendations available. Try different movies.")
    else:
        st.info("Select at least one movie to see recommendations.")

elif recommendation_type == "Content-Based":
    st.subheader("Content-Based Recommendations")
    st.markdown("""
    Based on genre similarity. Insights show overlapping genres and similarity scores.
    """)
    movie_list = movies['title'].tolist()
    selected_movies = st.multiselect("Select Movies You Like", movie_list)
    if selected_movies:
        recommended_insights = get_content_based_recommendations(selected_movies, similarity_matrix, movies)
        st.write("**Recommended Movies:**")
        if recommended_insights:
            for title, insight in recommended_insights:
                movie_id = title_to_id[title]
                movie_id_str = f"{movie_id}.0"
                imdb_url = "#"
                if movie_id in links['movieId'].values:
                    imdb_id = links[links['movieId'] == movie_id]['imdbId'].values[0]
                    imdb_url = f"https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/"
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if movie_id_str in poster_paths:
                        try:
                            st.image(poster_paths[movie_id_str], width=100)
                        except Exception:
                            st.write("Poster unavailable")
                        else:
                            st.write("No poster")
                with col2:
                    st.write(f"**{title}**")
                    overlapping_genres = ', '.join(insight['overlapping_genres'])
                    top_genres = ', '.join(insight['top_genres'])
                    st.write(f"- **Why Recommended**: Matches genres like {top_genres}.")
                    st.write(f"- **Overlapping Genres**: {overlapping_genres}")
                    st.write(f"- **Similarity Score**: {insight['similarity_score']:.2f}")
                    st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
        else:
            st.write("No recommendations available. Try different movies.")
    else:
        st.info("Select at least one movie to see recommendations.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and powered by the MovieLens dataset.")