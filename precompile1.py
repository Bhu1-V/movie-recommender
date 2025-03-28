import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

# Load movies data
movies = pd.read_csv('ml-latest-small/movies.csv')

# Compute genre-based similarity matrix
genres_list = movies['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(genres_list)
sim_matrix = cosine_similarity(genre_matrix).astype('float32')

# Save as a compressed sparse matrix
scipy.sparse.save_npz('similarity_matrix.npz', scipy.sparse.csr_matrix(sim_matrix))