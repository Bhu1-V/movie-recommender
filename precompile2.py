import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

# Load ratings data
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Sample 10% of the data (optional, if still desired)
ratings = ratings.sample(frac=0.1, random_state=42)

# Prepare data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train SVD model
model = SVD(n_factors=100)  # 100 latent factors is a good default
model.fit(trainset)

# Save the model and trainset
joblib.dump(model, 'svd_model.joblib')
joblib.dump(trainset, 'trainset.joblib')  # Needed for mapping IDs