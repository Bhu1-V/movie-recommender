# Movie Recommender System Project Report

## 1. Project Overview

### 1.1 Introduction
The Movie Recommender System is a web-based application built using Streamlit and deployed on Streamlit Cloud, designed to provide personalized and general movie recommendations to users. Leveraging the MovieLens small dataset, the system employs multiple recommendation techniques—Popularity-based, Collaborative Filtering (using SVD), and Content-Based Filtering—to cater to different user needs. This project integrates data science, artificial intelligence, and machine learning methodologies to deliver an efficient and scalable solution within the resource constraints of Streamlit Cloud's free tier (1GB memory, 50MB file size limit).

### 1.2 Objectives
- Develop a lightweight, deployable recommendation system optimized for limited resources.
- Implement three distinct recommendation algorithms to showcase versatility.
- Utilize precomputed models and matrices to reduce runtime computational overhead.
- Ensure all components (data, models, and files) fit within Streamlit Cloud’s constraints.

---

## 2. Data Description

### 2.1 Dataset
The project uses the **MovieLens Small Dataset** (downloaded from [GroupLens](https://grouplens.org/datasets/movielens/)), which includes:
- **ratings.csv**: 100,836 ratings from 610 users across 9,742 movies, with columns `userId`, `movieId`, `rating` (0.5 to 5.0, 0.5 increments), and `timestamp`.
- **movies.csv**: 9,742 movies with columns `movieId`, `title`, and `genres` (pipe-separated genre tags, e.g., "Action|Adventure").
- **links.csv**: 9,742 rows linking `movieId` to `imdbId` and `tmdbId` for external references.

### 2.2 Data Preprocessing
- **Sampling**: To optimize memory usage, the system uses a fixed 10% random sample of `ratings.csv` (~10,000 ratings), selected with `random_state=42` for reproducibility.
- **Column Selection**: Only essential columns are loaded (`userId`, `movieId`, `rating` from `ratings.csv`; `movieId`, `title`, `genres` from `movies.csv`; `movieId`, `imdbId` from `links.csv`) to minimize memory footprint.
- **Data Types**: Efficient types (`int32` for IDs, `float32` for ratings) reduce memory usage compared to default `int64`/`float64`.
- **Missing Values**: Rows with missing `userId`, `movieId`, or `rating` are dropped (though rare in this dataset).

---

## 3. Methodology

### 3.1 Recommendation Algorithms

#### 3.1.1 Popularity-Based Filtering
- **Approach**: Ranks movies using a weighted rating formula inspired by IMDb:
  $$
  \text{Weighted Rating} = \left( \frac{v}{v + m} \cdot R \right) + \left( \frac{m}{v + m} \cdot C \right)
  $$
  - $v$: Number of ratings for the movie.
  - $R$: Average rating for the movie.
  - $m$: Minimum ratings threshold (90th percentile of rating counts, ~10 ratings).
  - $C$: Mean rating across all movies (~3.5).
- **Implementation**: Computed using Pandas group-by operations on the sampled ratings dataset.
- **Output**: Top 10 movies by weighted rating, cached with `@st.cache_data`.
- **Complexity**: $O(n \log n)$ for sorting, where $n$ is the number of unique movies (~9,742).

#### 3.1.2 Collaborative Filtering (SVD)
- **Approach**: Uses Singular Value Decomposition (SVD) from the `surprise` library for matrix factorization, replacing the original KNN approach due to file size constraints (KNN’s 700MB vs. SVD’s ~4MB).
- **Theory**: 
  - Decomposes the user-movie rating matrix $R$ into:
  $$
  R \approx U \cdot \Sigma \cdot V^T
  $$
  - $U$: User latent factors (610 users $\times$ 100 factors).
  - $\Sigma$: Diagonal matrix of singular values.
  - $V^T$: Movie latent factors (9,742 movies $\times$ 100 factors).
- **Parameters**: `n_factors=100`, trained on the 10% sampled ratings (~10,000 entries).
- **Recommendation Logic**: 
  - Creates a pseudo-user with high ratings (5.0) for selected movies.
  - Predicts ratings for all other movies using SVD.
  - Returns top 10 movies by predicted rating.
- **Precomputation**: Trained locally and saved as `svd_model.joblib` (~4MB) and `trainset.joblib` (~5MB) using `joblib`.
- **Complexity**: Training is $O(n \cdot m \cdot k)$ where $n$ is users, $m$ is movies, and $k$ is factors; prediction is $O(m \cdot k)$.

#### 3.1.3 Content-Based Filtering
- **Approach**: Recommends movies based on genre similarity using cosine similarity:
  $$
  \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
  $$
- **Feature Engineering**: 
  - Genres are split from the `genres` column (e.g., "Action|Adventure") into lists.
  - `MultiLabelBinarizer` converts genres into a binary matrix (9,742 movies $\times$ 20 unique genres).
- **Similarity Matrix**: 
  - Originally a full 9,742 $\times$ 9,742 matrix (80MB as sparse `.npz`).
  - Optimized to store only top-100 similarities per movie, reducing size to ~4-5MB.
- **Recommendation Logic**: 
  - Computes average similarity scores for selected movies.
  - Excludes selected movies and returns top 10 by similarity.
- **Precomputation**: Computed locally and saved as `similarity_matrix.npz` using `scipy.sparse.csr_matrix`.
- **Complexity**: Precomputation is $O(n^2)$ for full matrix, reduced to $O(n \cdot k \cdot \log n)$ with top-K; inference is $O(n \cdot k)$.

### 3.2 Model Optimization for Deployment
- **Size Reduction**:
  - **KNN to SVD**: Original KNN model (700MB) replaced with SVD (~4MB) to fit Streamlit Cloud’s 50MB limit.
  - **Top-K Similarity**: Full similarity matrix (80MB) reduced to top-100 per movie (~4-5MB).
- **Precomputation**: Heavy computations (SVD training, similarity matrix) are performed locally and loaded at runtime to minimize memory usage on Streamlit Cloud.
- **Caching**: `@st.cache_data` and `@st.cache_resource` are used to cache data and models, avoiding redundant computation.

---

## 4. Technical Details

### 4.1 Tools and Libraries
- **Python**: Core programming language (v3.9+ recommended).
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical operations.
- **Scikit-Learn**: `MultiLabelBinarizer` and `cosine_similarity` for content-based filtering.
- **Surprise**: SVD implementation for collaborative filtering.
- **Scipy**: Sparse matrix handling (`csr_matrix`, `.npz` saving).
- **Joblib**: Model serialization.
- **Streamlit**: Web app framework.

### 4.2 Precomputation Scripts

#### SVD Model
```python
import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings = ratings.sample(frac=0.1, random_state=42)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD(n_factors=100)
model.fit(trainset)
joblib.dump(model, 'svd_model.joblib')
joblib.dump(trainset, 'trainset.joblib')
```
#### Similarity Matrix
``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import heapq

movies = pd.read_csv('ml-latest-small/movies.csv')
genres_list = movies['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(genres_list)
sim_matrix = cosine_similarity(genre_matrix).astype('float32')

def get_top_k_similarities(sim_matrix, k=100):
    rows, cols, data = [], [], []
    for i in range(sim_matrix.shape[0]):
        top_k_indices = heapq.nlargest(k + 1, range(len(sim_matrix[i])), key=sim_matrix[i].__getitem__)[1:]
        top_k_similarities = sim_matrix[i][top_k_indices]
        rows.extend([i] * k)
        cols.extend(top_k_indices)
        data.extend(top_k_similarities)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=sim_matrix.shape)

top_k_sim_matrix = get_top_k_similarities(sim_matrix, k=100)
scipy.sparse.save_npz('similarity_matrix.npz', top_k_sim_matrix)
```
## 4.3 File Sizes
- **svd_model.joblib**: ~4MB (SVD model with 100 factors).
- **trainset.joblib**: ~5MB (training set metadata).
- **similarity_matrix.npz**: ~4-5MB (top-100 similarities per movie).
- **ratings.csv, movies.csv, links.csv**: Kept in the `ml-latest-small/` folder (~3MB total compressed).

---

## 5. Implementation

### 5.1 App Structure
- **Frontend**: Streamlit UI with a sidebar for selecting recommendation type and multiselect for movie inputs.
- **Backend**: Loads precomputed SVD model and similarity matrix. Processes user inputs and generates recommendations.
- **Deployment**: Hosted on Streamlit Cloud with all files uploaded via GitHub.

### 5.2 Challenges and Solutions
- **Memory Limit (1GB)**: Reduced dataset to 10% and used efficient data types (`int32`, `float32`).
- **File Size Limit (50MB)**: Replaced KNN (700MB) with SVD (~4MB).
- **Reduced Similarity Matrix**: Optimized from 80MB to ~4-5MB with a top-K approach.
- **Runtime Efficiency**: Precomputed models and matrices to avoid on-the-fly computation.

---

## 6. Results and Evaluation

### 6.1 Performance
- **Popularity-Based**: Fastest (~seconds), as it’s a simple sort on preprocessed data.
- **SVD**: Predicts ratings in ~1-2 seconds for 10 recommendations, leveraging cached model.
- **Content-Based**: Retrieves top similarities in ~1 second with sparse matrix.

### 6.2 Quality
- **Popularity-Based**: High recall for broadly liked movies but lacks personalization.
- **SVD**: Good balance of personalization and generalization, validated by MovieLens benchmarks (RMSE ~0.9).
- **Content-Based**: Effective for genre-specific recommendations, though limited by genre-only features.

### 6.3 Deployment Success
Fits within Streamlit Cloud’s 1GB memory and 50MB file limits after optimization. No runtime crashes reported post-deployment.

---

## 7. Conclusion

### 7.1 Summary
The Movie Recommender System successfully integrates three recommendation techniques using data science and machine learning, optimized for a resource-constrained environment. SVD and top-K similarity approaches resolved initial file size issues, making the app deployable and efficient.

### 7.2 Future Work
- Incorporate additional features (e.g., movie tags, user demographics) for improved content-based filtering.
- Experiment with hybrid models combining SVD and content-based methods.
- Explore cloud-based model training to eliminate local precomputation.

---

## 8. References
- **MovieLens Dataset**: [GroupLens](https://grouplens.org/datasets/movielens/)
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **Surprise Library**: [Surprise Docs](http://surpriselib.com/)
- **Scikit-Learn**: [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
