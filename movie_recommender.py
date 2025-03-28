import pandas as pd
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, KNNBasic
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import json

print("Debug: Starting script execution")

# --- Load Poster Paths with Error Handling ---
print("Debug: Attempting to load posters.json")
try:
    with open('posters.json', 'r') as f:
        poster_paths = json.load(f)
    print("Debug: Successfully loaded posters.json")
except FileNotFoundError:
    poster_paths = {}
    st.error("**Error**: posters.json not found. Please ensure the file is in the correct directory.")
    print("Debug: posters.json not found, using empty dict")
except json.JSONDecodeError:
    poster_paths = {}
    st.error("**Error**: posters.json is corrupted. Please check the file format.")
    print("Debug: posters.json corrupted, using empty dict")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    print("Debug: Inside load_data function")
    print("Debug: Loading ratings.csv")
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    print("Debug: Loading movies.csv")
    movies = pd.read_csv('ml-latest-small/movies.csv')
    print("Debug: Loading links.csv")
    links = pd.read_csv('ml-latest-small/links.csv')
    print("Debug: Converting movieId columns to int")
    ratings['movieId'] = ratings['movieId'].astype(int)
    movies['movieId'] = movies['movieId'].astype(int)
    links['movieId'] = links['movieId'].astype(int)
    print("Debug: Computing movie stats")
    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    print("Debug: Computing rating distribution")
    rating_dist = ratings.groupby('movieId')['rating'].value_counts(normalize=True).unstack(fill_value=0)
    print("Debug: Returning data from load_data")
    return ratings, movies, links, movie_stats, rating_dist

print("Debug: Calling load_data")
ratings, movies, links, movie_stats, rating_dist = load_data()
print("Debug: load_data completed successfully")

# --- Popularity-Based Recommender ---
@st.cache_data
def compute_popularity_based_recommendations(ratings, movies):
    print("Debug: Inside compute_popularity_based_recommendations")
    print("Debug: Calculating global mean rating")
    C = ratings['rating'].mean()
    print("Debug: Computing number of ratings per movie")
    num_ratings = ratings.groupby('movieId').size()
    print("Debug: Calculating 90th percentile for minimum ratings")
    m = num_ratings.quantile(0.9)
    print("Debug: Aggregating movie stats")
    movie_stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
    movie_stats.columns = ['num_ratings', 'avg_rating']
    print("Debug: Filtering qualified movies")
    qualified_movies = movie_stats[movie_stats['num_ratings'] >= m].copy()
    print("Debug: Computing weighted ratings")
    qualified_movies['weighted_rating'] = (
        (qualified_movies['num_ratings'] / (qualified_movies['num_ratings'] + m)) * qualified_movies['avg_rating'] +
        (m / (qualified_movies['num_ratings'] + m)) * C
    )
    print("Debug: Sorting top movies")
    top_movies = qualified_movies.sort_values('weighted_rating', ascending=False).head(10)
    print("Debug: Merging with movie titles")
    top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId', how='left')
    print("Debug: Returning top movies")
    return top_movies

print("Debug: Calling compute_popularity_based_recommendations")
top_movies = compute_popularity_based_recommendations(ratings, movies)
print("Debug: compute_popularity_based_recommendations completed")

# --- Collaborative Filtering Recommender ---
@st.cache_data
def train_collaborative_filtering_model(ratings):
    print("Debug: Inside train_collaborative_filtering_model")
    print("Debug: Setting up Reader")
    reader = Reader(rating_scale=(0.5, 5.0))
    print("Debug: Loading data into surprise Dataset")
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    print("Debug: Building trainset")
    trainset = data.build_full_trainset()
    print("Debug: Configuring similarity options")
    sim_options = {'name': 'cosine', 'user_based': False}
    print("Debug: Initializing KNNBasic model")
    model = KNNBasic(sim_options=sim_options)
    print("Debug: Training model")
    model.fit(trainset)
    print("Debug: Returning trained model and trainset")
    return model, trainset

print("Debug: Calling train_collaborative_filtering_model")
model, trainset = train_collaborative_filtering_model(ratings)
print("Debug: train_collaborative_filtering_model completed")

# --- Content-Based Filtering ---
@st.cache_data
def compute_similarity_matrix(movies):
    print("Debug: Inside compute_similarity_matrix")
    print("Debug: Splitting genres into lists")
    genres_list = movies['genres'].apply(lambda x: x.split('|'))
    print("Debug: Initializing MultiLabelBinarizer")
    mlb = MultiLabelBinarizer()
    print("Debug: Transforming genres into matrix")
    genre_matrix = mlb.fit_transform(genres_list)
    print("Debug: Computing cosine similarity")
    sim_matrix = cosine_similarity(genre_matrix)
    print("Debug: Returning similarity matrix")
    return sim_matrix

print("Debug: Calling compute_similarity_matrix")
similarity_matrix = compute_similarity_matrix(movies)
print("Debug: compute_similarity_matrix completed")

def get_content_based_recommendations(selected_titles, similarity_matrix, movies, k=10):
    print("Debug: Inside get_content_based_recommendations")
    print("Debug: Finding indices for selected titles")
    selected_indices = [movies[movies['title'] == title].index[0] for title in selected_titles if title in movies['title'].values]
    if not selected_indices:
        print("Debug: No valid titles selected, returning empty list")
        return []
    print("Debug: Computing average similarity")
    avg_similarity = np.mean(similarity_matrix[selected_indices], axis=0)
    print("Debug: Zeroing out selected movies")
    avg_similarity[selected_indices] = 0
    print("Debug: Sorting top similar movies")
    top_indices = np.argsort(avg_similarity)[::-1][:k]
    insights = []
    print("Debug: Generating insights for recommendations")
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
    print("Debug: Returning content-based recommendations")
    return insights

# Function for collaborative filtering with detailed insights
def get_recommendations(selected_ids, model, trainset, ratings, k=10):
    print("Debug: Inside get_recommendations")
    recommended = {}
    for movie_id in selected_ids:
        print(f"Debug: Processing movie_id {movie_id}")
        try:
            inner_id = trainset.to_inner_iid(movie_id)
            print(f"Debug: Getting neighbors for inner_id {inner_id}")
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
            print(f"Debug: ValueError for movie_id {movie_id}, skipping")
            continue
    print("Debug: Returning collaborative filtering recommendations")
    return recommended

# Create mappings between movie titles and IDs
print("Debug: Creating title-to-ID mappings")
title_to_id = dict(zip(movies['title'], movies['movieId']))
id_to_title = dict(zip(movies['movieId'], movies['title']))
print("Debug: Mappings created successfully")

# # --- Streamlit Web App ---
# print("Debug: Setting up main UI")
# st.title("Movie Recommender System")
# st.markdown("""
# Welcome to the Movie Recommender System! Choose a recommendation type from the sidebar:
# - **Popularity-based**: See the top 10 movies based on weighted ratings.
# - **Collaborative Filtering**: Get personalized recommendations based on user rating patterns.
# - **Content-Based**: Get recommendations based on genre similarities to the movies you select.
# """)

# print("Debug: Adding sidebar selectbox")
# recommendation_type = st.sidebar.selectbox(
#     "Choose Recommendation Type",
#     ["Popularity-based", "Collaborative Filtering", "Content-Based"]
# )

# if recommendation_type == "Popularity-based":
#     print("Debug: Rendering Popularity-based section")
#     st.subheader("Top 10 Popular Movies")
#     st.markdown("""
#     These movies are ranked using a weighted rating formula:  
#     `(v/(v+m) * R) + (m/(v+m) * C)`, where `v` is the number of ratings, `R` is the average rating,  
#     `m` is the 90th percentile of ratings, and `C` is the mean rating across all movies.
#     """)
#     for idx, row in top_movies.iterrows():
#         print(f"Debug: Processing popularity movie {row['title']}")
#         movie_id = row['movieId']
#         movie_id_str = f"{movie_id}.0"
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             if movie_id_str in poster_paths:
#                 try:
#                     st.image(poster_paths[movie_id_str], width=100)
#                     print(f"Debug: Loaded poster for {movie_id}")
#                 except Exception:
#                     st.write("Poster unavailable")
#                     print(f"Debug: Failed to load poster for {movie_id}")
#             else:
#                 st.write("No poster")
#                 print(f"Debug: No poster found for {movie_id}")
#         with col2:
#             st.write(f"**{row['title']}** - Weighted Rating: {row['weighted_rating']:.2f}")
#             print(f"Debug: Displayed title and rating for {row['title']}")

# elif recommendation_type == "Collaborative Filtering":
#     print("Debug: Rendering Collaborative Filtering section")
#     st.subheader("Personalized Recommendations (Collaborative Filtering)")
#     st.markdown("""
#     These recommendations are based on patterns in user ratings. Movies are suggested because users who liked the movies you selected also liked these.  
#     Insights include similarity scores, user overlap, genres, and rating details.
#     """)
#     movie_list = movies['title'].tolist()
#     print("Debug: Displaying movie selection multiselect")
#     selected_movies = st.multiselect("Select Movies You Like", movie_list)
#     if selected_movies:
#         print("Debug: Processing selected movies")
#         selected_ids = [title_to_id[title] for title in selected_movies if title in title_to_id]
#         print("Debug: Calling get_recommendations")
#         recommended_dict = get_recommendations(selected_ids, model, trainset, ratings, k=10)
#         recommended_ids = list(recommended_dict.keys())
#         recommended_titles = [id_to_title[mid] for mid in recommended_ids if mid in id_to_title]
        
#         st.write("**Recommended Movies:**")
#         if recommended_titles:
#             for title in recommended_titles:
#                 print(f"Debug: Processing collaborative movie {title}")
#                 movie_id = title_to_id[title]
#                 movie_id_str = f"{movie_id}.0"
#                 imdb_id = links[links['movieId'] == movie_id]['imdbId'].values[0]
#                 imdb_url = f"https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/"
                
#                 col1, col2 = st.columns([1, 3])
#                 with col1:
#                     if movie_id_str in poster_paths:
#                         try:
#                             st.image(poster_paths[movie_id_str], width=100)
#                             print(f"Debug: Loaded poster for {movie_id}")
#                         except Exception:
#                             st.write("Poster unavailable")
#                             print(f"Debug: Failed to load poster for {movie_id}")
#                     else:
#                         st.write("No poster")
#                         print(f"Debug: No poster found for {movie_id}")
#                 with col2:
#                     st.write(f"**{title}**")
#                     similar_to = recommended_dict[movie_id]['similar_to']
#                     sim_score = recommended_dict[movie_id]['sim_score']
#                     overlap_pct = recommended_dict[movie_id]['overlap_pct']
#                     avg_rating = movie_stats.loc[movie_id, 'avg_rating'] if movie_id in movie_stats.index else 'N/A'
#                     num_ratings = movie_stats.loc[movie_id, 'num_ratings'] if movie_id in movie_stats.index else 'N/A'
#                     genres_str = movies[movies['movieId'] == movie_id]['genres'].values[0]
#                     rating_5_pct = rating_dist.loc[movie_id, 5.0] * 100 if movie_id in rating_dist.index and 5.0 in rating_dist.columns else 0
                    
#                     st.write(f"- **Why Recommended**: Similar to '{similar_to}' with a similarity score of {sim_score:.2f} (0 to 1).")
#                     st.write(f"- **User Overlap**: Liked by {overlap_pct:.1f}% of users who rated '{similar_to}'.")
#                     st.write(f"- **Genres**: {genres_str.replace('|', ', ')}")
#                     st.write(f"- **Rating Details**: Avg {avg_rating:.2f} from {num_ratings} ratings; {rating_5_pct:.1f}% gave 5 stars" if avg_rating != 'N/A' else "- **Rating Details**: Not available")
#                     st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
#                     print(f"Debug: Displayed details for {title}")
#         else:
#             st.write("No recommendations available. Try selecting different movies.")
#             print("Debug: No collaborative recommendations available")
#     else:
#         st.info("Please select at least one movie to see recommendations.")
#         print("Debug: No movies selected for collaborative filtering")

# elif recommendation_type == "Content-Based":
#     print("Debug: Rendering Content-Based section")
#     st.subheader("Content-Based Recommendations")
#     st.markdown("""
#     These recommendations are based on the similarity of movie genres. The 'Similarity Score' shows how closely the recommended movie's genres match those of your selected movies.  
#     Insights include overlapping genres and top matching genres.
#     """)
#     movie_list = movies['title'].tolist()
#     print("Debug: Displaying movie selection multiselect")
#     selected_movies = st.multiselect("Select Movies You Like", movie_list)
#     if selected_movies:
#         print("Debug: Processing selected movies for content-based")
#         recommended_insights = get_content_based_recommendations(selected_movies, similarity_matrix, movies)
#         st.write("**Recommended Movies:**")
#         if recommended_insights:
#             for title, insight in recommended_insights:
#                 print(f"Debug: Processing content-based movie {title}")
#                 movie_id = title_to_id[title]
#                 movie_id_str = f"{movie_id}.0"
#                 imdb_id = links[links['movieId'] == movie_id]['imdbId'].values[0]
#                 imdb_url = f"https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/"
                
#                 col1, col2 = st.columns([1, 3])
#                 with col1:
#                     if movie_id_str in poster_paths:
#                         try:
#                             st.image(poster_paths[movie_id_str], width=100)
#                             print(f"Debug: Loaded poster for {movie_id}")
#                         except Exception:
#                             st.write("Poster unavailable")
#                             print(f"Debug: Failed to load poster for {movie_id}")
#                     else:
#                         st.write("No poster")
#                         print(f"Debug: No poster found for {movie_id}")
#                 with col2:
#                     st.write(f"**{title}**")
#                     overlapping_genres = ', '.join(insight['overlapping_genres'])
#                     top_genres = ', '.join(insight['top_genres'])
#                     st.write(f"- **Why Recommended**: Matches genres like {top_genres} with your selections.")
#                     st.write(f"- **Overlapping Genres**: {overlapping_genres}")
#                     st.write(f"- **Similarity Score**: {insight['similarity_score']:.2f} (0 to 1)")
#                     st.markdown(f'<a href="{imdb_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:5px 10px;border:none;border-radius:5px;cursor:pointer;">View on IMDb</button></a>', unsafe_allow_html=True)
#                     print(f"Debug: Displayed details for {title}")
#         else:
#             st.write("No recommendations available. Try selecting different movies.")
#             print("Debug: No content-based recommendations available")
#     else:
#         st.info("Please select at least one movie to see recommendations.")
#         print("Debug: No movies selected for content-based")

# # Footer
# print("Debug: Rendering footer")
# st.markdown("---")
# st.markdown("Built with Streamlit and powered by the MovieLens dataset.")
# print("Debug: Script execution completed")