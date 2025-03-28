import pandas as pd
from surprise import Dataset, Reader, KNNBasic
import joblib

# Load ratings data
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Prepare data
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train the model
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Save the model and trainset
joblib.dump(model, 'knn_model.joblib')
joblib.dump(trainset, 'trainset.joblib')