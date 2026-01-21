import pandas as pd
from pymongo import MongoClient
import pickle
import os

# ==========================================
# MongoDB Configuration
# ==========================================
client = MongoClient("mongodb://localhost:27017/")
db = client["movie_rec_system"]

def init_database():
    print("Starting database initialization...")
    
    # 1. Load Data
    movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("data/ml-latest-small/ratings.csv")
    
    # Load model mapping to ensure consistency
    if not os.path.exists("model_final.pkl"):
        print("Error: model_final.pkl not found! Please train your model first.")
        return
        
    with open("model_final.pkl", "rb") as f:
        model_data = pickle.load(f)
        u_map = model_data.get("u_map", {})
        i_map = model_data.get("i_map", {})

    # 2. Process Movie Statistics (for Cold Start)
    print("Calculating movie statistics...")
    movie_stats = ratings_df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    # 3. Import Movies
    print("Importing movies to MongoDB...")
    db.movies.drop() # Clear existing
    movies_to_insert = []
    for _, row in movies_df.iterrows():
        mid = int(row['movieId'])
        stats = movie_stats[movie_stats['movieId'] == mid]
        
        movie_doc = {
            "movieId": mid,
            "title": row['title'],
            "genres": row['genres'].split('|'),
            "inner_idx": i_map.get(mid, -1), # Index in model Q matrix
            "avg_rating": float(stats['avg_rating'].values[0]) if not stats.empty else 0.0,
            "rating_count": int(stats['rating_count'].values[0]) if not stats.empty else 0
        }
        movies_to_insert.append(movie_doc)
    db.movies.insert_many(movies_to_insert)

    # 4. Import Old Users (from ratings and u_map)
    print("Initializing old users (Password: 000000)...")
    db.users.drop()
    users_to_insert = []
    all_user_ids = set(ratings_df['userId'].unique()).union(set(u_map.keys()))
    
    for uid in all_user_ids:
        user_doc = {
            "userId": int(uid),
            "username": f"user_{uid}",
            "password": "000000", # Requirement: Old users use 000000
            "inner_idx": u_map.get(uid, -1), # Index in model P matrix
            "is_new": False
        }
        users_to_insert.append(user_doc)
    db.users.insert_many(users_to_insert)

    # 5. Import Ratings
    print("Importing ratings...")
    db.ratings.drop()
    db.ratings.insert_many(ratings_df.to_dict('records'))

    print("Database initialization complete!")

if __name__ == "__main__":
    init_database()