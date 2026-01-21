import numpy as np
import pickle
import os
import pandas as pd

class MovieRecommender:
    def __init__(self, model_path="model_final.pkl", ratings_path="data/ml-latest-small/ratings.csv"):
        self.model_path = model_path
        self.is_loaded = False
        self.load_model(ratings_path)

    def load_model(self, ratings_path):
        if not os.path.exists(self.model_path):
            print("Model file not found.")
            return

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.P = data['P']
            self.Q = data['Q']
            self.B_u = data['B_u']
            self.B_i = data['B_i']
            self.u_map = data['u_map']
            self.i_map = data['i_map']
            
            # Inverse map for getting raw IDs from inner index
            self.inv_i_map = {v: k for k, v in self.i_map.items()}
            
            # Handle missing global_mean by calculating it from data/ml-latest-small/ratings.csv
            if 'global_mean' in data:
                self.global_mean = data['global_mean']
            else:
                print("Global mean not found in pkl, calculating from CSV...")
                df = pd.read_csv(ratings_path)
                self.global_mean = df['rating'].mean()
        
        self.is_loaded = True
        print(f"Model loaded. Global Mean: {self.global_mean:.4f}")

    def predict_score(self, u_idx, i_idx):
        """
        Manually implement the prediction logic: Mean + Bu + Bi + P·Q
        Strictly following the train.py logic.
        """
        dot_product = np.dot(self.P[u_idx], self.Q[i_idx])
        score = self.global_mean + self.B_u[u_idx] + self.B_i[i_idx] + dot_product
        return max(0.5, min(5.0, score))

    def get_personalized_recommendations(self, user_id, top_n=10, watched_ids=[]):
        """
        Generate recommendations for existing users using the latent factors.
        """
        if user_id not in self.u_map:
            return []

        u_idx = self.u_map[user_id]
        scores = []

        # Predict for all movies in the model
        # Optimization: We can use matrix multiplication for speed
        # All scores = mean + Bu[u_idx] + B_i_array + dot(P[u_idx], Q.T)
        dot_products = np.dot(self.P[u_idx], self.Q.T)
        all_predicted_scores = self.global_mean + self.B_u[u_idx] + self.B_i + dot_products

        for i_idx, score in enumerate(all_predicted_scores):
            raw_movie_id = self.inv_i_map.get(i_idx)
            if raw_movie_id and raw_movie_id not in watched_ids:
                scores.append((raw_movie_id, float(score)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def get_similar_movies(self, movie_id, top_n=5):
        """
        Manually implement cosine similarity using NumPy.
        Formula: (A·B) / (||A|| * ||B||)
        """
        if movie_id not in self.i_map:
            return []

        target_idx = self.i_map[movie_id]
        target_vec = self.Q[target_idx]
        
        # Calculate norms for all movie vectors
        norms = np.linalg.norm(self.Q, axis=1)
        target_norm = np.linalg.norm(target_vec)
        
        # Cosine similarity calculation
        similarities = np.dot(self.Q, target_vec) / (norms * target_norm + 1e-9)
        
        results = []
        for i_idx, sim in enumerate(similarities):
            raw_id = self.inv_i_map.get(i_idx)
            if raw_id and raw_id != movie_id:
                results.append((raw_id, float(sim)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]