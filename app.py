from flask import Flask, request, jsonify, session, render_template
from pymongo import MongoClient
from recommender import MovieRecommender
import datetime

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["movie_rec_system"]

# Initialize Recommender Engine
engine = MovieRecommender()

# ==========================================
# Page Routes
# ==========================================
@app.route('/')
def index():
    # Render the main frontend page
    return render_template('index.html')

@app.route('/my-ratings')
def my_ratings_page():
    # Render the My Ratings page
    return render_template('my_ratings.html')

# ==========================================
# Auth Routes
# ==========================================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if db.users.find_one({"username": username}):
        return jsonify({"success": False, "message": "Username already exists"}), 400
    
    # Calculate new ID (Current max + 1)
    new_user_id = db.users.count_documents({}) + 1000 
    db.users.insert_one({
        "userId": new_user_id,
        "username": username,
        "password": password,
        "inner_idx": -1, # Flag for cold start
        "is_new": True,
        "created_at": datetime.datetime.now()
    })
    return jsonify({"success": True, "userId": new_user_id})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = db.users.find_one({"username": username, "password": password})
    if user:
        session['user_id'] = user['userId']
        return jsonify({"success": True, "userId": user['userId'], "username": username})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/check-session', methods=['GET'])
def check_session():
    user_id = session.get('user_id')
    if user_id:
        user = db.users.find_one({"userId": user_id})
        if user:
            return jsonify({"success": True, "userId": user['userId'], "username": user['username']})
    return jsonify({"success": False}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

# ==========================================
# Movie Routes
# ==========================================
@app.route('/api/recommend', methods=['GET'])
def recommend():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Please login"}), 401
    
    user = db.users.find_one({"userId": user_id})
    # Get watched movies from ratings
    watched = [r['movieId'] for r in db.ratings.find({"userId": user_id})]
    
    # 1. Personalized Recommendation (Existing Users)
    if user.get('inner_idx') != -1:
        recs = engine.get_personalized_recommendations(user_id, watched_ids=watched)
        movie_ids = [r[0] for r in recs]
        movies = list(db.movies.find({"movieId": {"$in": movie_ids}}, {"_id": 0}))
        # Ensure order matches recommendation score
        movies.sort(key=lambda x: movie_ids.index(x['movieId']))
        return jsonify({"type": "personalized", "data": movies})
    
    # 2. Cold Start Recommendation (New Users)
    else:
        # Get high rated and popular movies (Hot recommendation)
        hot_movies = list(db.movies.find({}, {"_id": 0}).sort([("avg_rating", -1), ("rating_count", -1)]).limit(10))
        return jsonify({"type": "cold_start", "data": hot_movies})

@app.route('/api/movie/<int:mid>', methods=['GET'])
def movie_detail(mid):
    # Fetch specific movie and find similar ones via vector similarity
    movie = db.movies.find_one({"movieId": mid}, {"_id": 0})
    if not movie:
        return jsonify({"error": "Movie not found"}), 404
        
    # Get similar movies using latent factors (Item-based filtering)
    similar_res = engine.get_similar_movies(mid)
    similar_ids = [s[0] for s in similar_res]
    similar_movies = list(db.movies.find({"movieId": {"$in": similar_ids}}, {"_id": 0}))
    
    return jsonify({
        "movie": movie,
        "similar": similar_movies
    })

@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    page = int(request.args.get('page', 1))
    limit = 20
    movies = list(db.movies.find({}, {"_id": 0}).skip((page-1)*limit).limit(limit))
    return jsonify(movies)

@app.route('/api/my-ratings', methods=['GET'])
def my_ratings():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Please login"}), 401
    
    # Get all ratings by this user
    user_ratings = list(db.ratings.find({"userId": user_id}))
    movie_ids = [r['movieId'] for r in user_ratings]
    
    # Get movie details
    movies = list(db.movies.find({"movieId": {"$in": movie_ids}}, {"_id": 0}))
    
    # Merge rating info with movie info
    result = []
    for movie in movies:
        user_rating = next((r['rating'] for r in user_ratings if r['movieId'] == movie['movieId']), None)
        result.append({
            **movie,
            "user_rating": user_rating
        })
    
    # Sort by user rating (highest first)
    result.sort(key=lambda x: x['user_rating'], reverse=True)
    return jsonify(result)

@app.route('/api/rate', methods=['POST'])
def rate_movie():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login"}), 401
    
    data = request.json
    movie_id = data.get('movieId')
    rating = data.get('rating')
    
    if not movie_id or not rating:
        return jsonify({"success": False, "message": "Missing movieId or rating"}), 400
    
    if rating < 0.5 or rating > 5.0:
        return jsonify({"success": False, "message": "Rating must be between 0.5 and 5.0"}), 400
    
    # Check if user already rated this movie
    existing_rating = db.ratings.find_one({"userId": user_id, "movieId": movie_id})
    
    if existing_rating:
        # Update existing rating
        db.ratings.update_one(
            {"userId": user_id, "movieId": movie_id},
            {"$set": {"rating": rating, "timestamp": datetime.datetime.now().timestamp()}}
        )
    else:
        # Insert new rating
        db.ratings.insert_one({
            "userId": user_id,
            "movieId": movie_id,
            "rating": rating,
            "timestamp": datetime.datetime.now().timestamp()
        })
    
    # Recalculate movie statistics
    movie_ratings = list(db.ratings.find({"movieId": movie_id}))
    avg_rating = sum(r['rating'] for r in movie_ratings) / len(movie_ratings)
    rating_count = len(movie_ratings)
    
    db.movies.update_one(
        {"movieId": movie_id},
        {"$set": {"avg_rating": avg_rating, "rating_count": rating_count}}
    )
    
    return jsonify({"success": True, "avg_rating": avg_rating, "rating_count": rating_count})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)