# movie_recommender.py

import pandas as pd
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, KNNBasic
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- Load Poster Paths with Error Handling ---
try:
    with open('posters.json', 'r') as f:
        poster_paths = json.load(f)
except FileNotFoundError:
    poster_paths = {}
    st.error("**Error**: posters.json not found. Please ensure the file is in the correct directory.")
except json.JSONDecodeError:
    poster_paths = {}
    st.error("**Error**: posters.json is corrupted. Please check the file format.")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Load and preprocess the MovieLens dataset."""
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')
    links = pd.read_csv('ml-latest-small/links.csv')
    ratings['movieId'] = ratings['movieId'].astype(int)
    movies['movieId'] = movies['movieId'].astype(int)
    links['movieId'] = links['movieId'].astype(int)
    # Keep genres as strings (do not split here)
    # Removed: movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    # Compute movie stats
    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    # Rating distribution
    rating_dist = ratings.groupby('movieId')['rating'].value_counts(normalize=True).unstack(fill_value=0)
    return ratings, movies, links, movie_stats, rating_dist

ratings, movies, links, movie_stats, rating_dist = load_data()

# --- Popularity-Based Recommender ---
@st.cache_data
def compute_popularity_based_recommendations(ratings, movies):
    """Compute top 10 movies based on weighted ratings."""
    C = ratings['rating'].mean()
    num_ratings = ratings.groupby('movieId').size()
    m = num_ratings.quantile(0.9)
    movie_stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
    movie_stats.columns = ['num_ratings', 'avg_rating']
    qualified_movies = movie_stats[movie_stats['num_ratings'] >= m].copy()
    qualified_movies['weighted_rating'] = (
        (qualified_movies['num_ratings'] / (qualified_movies['num_ratings'] + m)) * qualified_movies['avg_rating'] +
        (m / (qualified_movies['num_ratings'] + m)) * C
    )
    top_movies = qualified_movies.sort_values('weighted_rating', ascending=False).head(10)
    top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId', how='left')
    return top_movies

top_movies = compute_popularity_based_recommendations(ratings, movies)

# --- Collaborative Filtering Recommender ---
@st.cache_data
def train_collaborative_filtering_model(ratings):
    """Train an item-based collaborative filtering model."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model, trainset

model, trainset = train_collaborative_filtering_model(ratings)

# --- Content-Based Filtering ---
@st.cache_data
def compute_similarity_matrix(movies):
    """Compute cosine similarity matrix based on movie genres."""
    # Split genres inside the function
    genres_list = movies['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres_list)
    return cosine_similarity(genre_matrix)

similarity_matrix = compute_similarity_matrix(movies)

def get_content_based_recommendations(selected_titles, similarity_matrix, movies, k=10):
    """Generate content-based recommendations with insights."""
    selected_indices = [movies[movies['title'] == title].index[0] for title in selected_titles if title in movies['title'].values]
    if not selected_indices:
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
            'similarity_score': similarity_score,
            'top_genres': top_genres
        }
        insights.append((title, insight))
    return insights

# Function for collaborative filtering with detailed insights
def get_recommendations(selected_ids, model, trainset, ratings, k=10):
    """Generate collaborative filtering recommendations with insights."""
    recommended = {}
    for movie_id in selected_ids:
        try:
            inner_id = trainset.to_inner_iid(movie_id)
            neighbors = model.get_neighbors(inner_id, k=50)
            for neighbor in neighbors:
                raw_id = trainset.to_raw_iid(neighbor)
                if raw_id not in selected_ids and raw_id not in recommended:
                    sim_score = model.sim[inner_id, neighbor]
                    selected_users = set(ratings[ratings['movieId'] == movie_id]['userId'])
                    rec_users = set(ratings[ratings['movieId'] == raw_id]['userId'])
                    overlap = len(selected_users.intersection(rec_users))
                    total_users = len(selected_users)
                    overlap_pct = (overlap / total_users) * 100 if total_users > 0 else 0
                    recommended[raw_id] = {
                        'similar_to': id_to_title[movie_id],
                        'sim_score': sim_score,
                        'overlap_pct': overlap_pct
                    }
                    if len(recommended) >= k:
                        break
            if len(recommended) >= k:
                break
        except ValueError:
            continue
    return recommended

# Create mappings between movie titles and IDs
title_to_id = dict(zip(movies['title'], movies['movieId']))
id_to_title = dict(zip(movies['movieId'], movies['title']))

# --- Streamlit Web App ---
st.title("Movie Recommender System")
st.markdown("""
Welcome to the Movie Recommender System! Choose a recommendation type from the sidebar:
- **Popularity-based**: See the top 10 movies based on weighted ratings.
- **Collaborative Filtering**: Get personalized recommendations based on user rating patterns.
- **Content-Based**: Get recommendations based on genre similarities to the movies you select.
""")

recommendation_type = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["Popularity-based", "Collaborative Filtering", "Content-Based"]
)

if recommendation_type == "Popularity-based":
    st.subheader("Top 10 Popular Movies")
    st.markdown("""
    These movies are ranked using a weighted rating formula:  
    `(v/(v+m) * R) + (m/(v+m) * C)`, where `v` is the number of ratings, `R` is the average rating,  
    `m` is the 90th percentile of ratings, and `C` is the mean rating across all movies.
    """)
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

elif recommendation_type == "Collaborative Filtering":
    st.subheader("Personalized Recommendations (Collaborative Filtering)")
    st.markdown("""
    These recommendations are based on patterns in user ratings. Movies are suggested because users who liked the movies you selected also liked these.  
    Insights include similarity scores, user overlap, genres, and rating details.
    """)
    movie_list = movies['title'].tolist()
    selected_movies = st.multiselect("Select Movies You Like", movie_list)
    if selected_movies:
        selected_ids = [title_to_id[title] for title in selected_movies if title in title_to_id]
        recommended_dict = get_recommendations(selected_ids, model, trainset, ratings, k=10)
        recommended_ids = list(recommended_dict.keys())
        recommended_titles = [id_to_title[mid] for mid in recommended_ids if mid in id_to_title]
        
        st.write("**Recommended Movies:**")
        if recommended_titles:
            for title in recommended_titles:
                movie_id = title_to_id[title]
                movie_id_str = f"{movie_id}.0"
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
                    similar_to = recommended_dict[movie_id]['similar_to']
                    sim_score = recommended_dict[movie_id]['sim_score']
                    overlap_pct = recommended_dict[movie_id]['overlap_pct']
                    avg_rating = movie_stats.loc[movie_id, 'avg_rating'] if movie_id in movie_stats.index else 'N/A'
                    num_ratings = movie_stats.loc[movie_id, 'num_ratings'] if movie_id in movie_stats.index else 'N/A'
                    genres_str = movies[movies['movieId'] == movie_id]['genres'].values[0]
                    rating_5_pct = rating_dist.loc[movie_id, 5.0] * 100 if movie_id in rating_dist.index and 5.0 in rating_dist.columns else 0
                    
                    st.write(f"- **Why Recommended**: Similar to '{similar_to}' with a similarity score of {sim_score:.2f} (0 to 1).")
                    st.write(f"- **User Overlap**: Liked by {overlap_pct:.1f}% of users who rated '{similar_to}'.")
                    st.write(f"- **Genres**: {genres_str.replace('|', ', ')}")
                    st.write(f"- **Rating Details**: Avg {avg_rating:.2f} from {num_ratings} ratings; {rating_5_pct:.1f}% gave 5 stars" if avg_rating != 'N/A' else "- **Rating Details**: Not available")
                    st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
        else:
            st.write("No recommendations available. Try selecting different movies.")
    else:
        st.info("Please select at least one movie to see recommendations.")

elif recommendation_type == "Content-Based":
    st.subheader("Content-Based Recommendations")
    st.markdown("""
    These recommendations are based on the similarity of movie genres. The 'Similarity Score' shows how closely the recommended movie's genres match those of your selected movies.  
    Insights include overlapping genres and top matching genres.
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
                    st.write(f"- **Why Recommended**: Matches genres like {top_genres} with your selections.")
                    st.write(f"- **Overlapping Genres**: {overlapping_genres}")
                    st.write(f"- **Similarity Score**: {insight['similarity_score']:.2f} (0 to 1)")
                    st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
        else:
            st.write("No recommendations available. Try selecting different movies.")
    else:
        st.info("Please select at least one movie to see recommendations.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and powered by the MovieLens dataset.")