import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import heapq

# Load movies data
movies = pd.read_csv('ml-latest-small/movies.csv')

# Compute genre-based similarity matrix
genres_list = movies['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(genres_list)
sim_matrix = cosine_similarity(genre_matrix).astype('float32')

# Function to extract top-K similarities for each movie
def get_top_k_similarities(sim_matrix, k=100):
    rows, cols, data = [], [], []
    for i in range(sim_matrix.shape[0]):
        # Get top-K indices and similarities (excluding self-similarity at i)
        top_k_indices = heapq.nlargest(k + 1, range(len(sim_matrix[i])), key=sim_matrix[i].__getitem__)[1:]  # Skip self
        top_k_similarities = sim_matrix[i][top_k_indices]
        rows.extend([i] * k)
        cols.extend(top_k_indices)
        data.extend(top_k_similarities)
    # Create sparse matrix
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=sim_matrix.shape)

# Compute top-K similarities (e.g., K=100)
top_k_sim_matrix = get_top_k_similarities(sim_matrix, k=100)

# Save as a compressed sparse matrix
scipy.sparse.save_npz('similarity_matrix.npz', top_k_sim_matrix)