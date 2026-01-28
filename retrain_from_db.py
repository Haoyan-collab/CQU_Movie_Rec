"""
Retrain Model Script
====================
This script extracts ratings from MongoDB and retrains the ALS model.
It merges original CSV data with new user ratings from the database.

Usage:
    python retrain_from_db.py [--db-only]
    
    --db-only: Only use ratings from MongoDB (ignore CSV)
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark import SparkConf, StorageLevel

# ==========================================
# Configuration (Same as train.py)
# ==========================================
DRIVER_MEMORY = "2g"
EXECUTOR_MEMORY = "4g"
PARALLELISM = 8
RANK = 20
ITERATIONS = 20
LAMBDA_REG = 0.1
BIAS_REG = 10

def get_spark_session():
    conf = SparkConf() \
        .setAppName("MovieLens_ALS_Retrain") \
        .setMaster("local[*]") \
        .set("spark.driver.memory", DRIVER_MEMORY) \
        .set("spark.executor.memory", EXECUTOR_MEMORY) \
        .set("spark.default.parallelism", str(PARALLELISM)) \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    return SparkSession.builder.config(conf=conf).getOrCreate()

# ==========================================
# Data Extraction from MongoDB
# ==========================================
def extract_ratings_from_db(mongo_uri="mongodb://localhost:27017/", db_name="movie_rec_system"):
    """
    Extract all ratings from MongoDB and return as DataFrame
    """
    print(f"[{time.strftime('%H:%M:%S')}] Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    # Get all ratings
    ratings_cursor = db.ratings.find({}, {"_id": 0, "userId": 1, "movieId": 1, "rating": 1})
    ratings_list = list(ratings_cursor)
    
    if not ratings_list:
        print("Warning: No ratings found in MongoDB!")
        return pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    
    df = pd.DataFrame(ratings_list)
    print(f"[{time.strftime('%H:%M:%S')}] Extracted {len(df)} ratings from MongoDB")
    return df

def merge_ratings_data(csv_path="data/ml-latest-small/ratings.csv", use_db_only=False):
    """
    Merge original CSV data with MongoDB ratings
    """
    db_df = extract_ratings_from_db()
    
    if use_db_only:
        print(f"[{time.strftime('%H:%M:%S')}] Using only MongoDB data...")
        combined_df = db_df
    else:
        # Load original CSV data
        if os.path.exists(csv_path):
            print(f"[{time.strftime('%H:%M:%S')}] Loading original CSV data...")
            csv_df = pd.read_csv(csv_path)[['userId', 'movieId', 'rating']]
            print(f"[{time.strftime('%H:%M:%S')}] Original CSV: {len(csv_df)} ratings")
            
            # Merge: Keep original CSV + new ratings from DB (avoid duplicates)
            # Strategy: Use CSV as base, update with DB ratings (DB overrides CSV for same user-movie pairs)
            combined_df = pd.concat([csv_df, db_df], ignore_index=True)
            
            # Remove duplicates, keeping the last occurrence (DB data is more recent)
            combined_df = combined_df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
            print(f"[{time.strftime('%H:%M:%S')}] Merged data: {len(combined_df)} ratings")
        else:
            print(f"Warning: CSV file not found at {csv_path}, using only MongoDB data")
            combined_df = db_df
    
    return combined_df

# ==========================================
# RMSE Calculation (Same as train.py)
# ==========================================
def compute_rmse(rdd, P_bd, Q_bd, Bu_bd, Bi_bd, global_mean):
    P = P_bd.value
    Q = Q_bd.value
    Bu = Bu_bd.value
    Bi = Bi_bd.value
    
    def predict_error(row):
        u, i, r = row
        pred = global_mean
        if u < len(Bu): pred += Bu[u]
        if i < len(Bi): pred += Bi[i]
        if u < len(P) and i < len(Q):
            pred += np.dot(P[u], Q[i])
        pred = max(0.5, min(5.0, pred))
        return (r - pred) ** 2

    mse = rdd.map(predict_error).mean()
    return np.sqrt(mse)

# ==========================================
# Core Solver (Same as train.py)
# ==========================================
def solve_batch_residual(iterator, fixed_vecs_bd, fixed_biases_bd, my_biases_bd, global_mean, rank, lambda_reg):
    fixed_vecs = fixed_vecs_bd.value
    fixed_biases = fixed_biases_bd.value
    my_biases = my_biases_bd.value
    
    eye = np.eye(rank) * lambda_reg
    results = []
    
    for _id, ratings in iterator:
        if not ratings: continue
        
        target_ids = [r[0] for r in ratings]
        raw_ratings = np.array([r[1] for r in ratings], dtype=np.float32)
        
        relevant_biases = fixed_biases[target_ids]
        
        if 0 <= _id < len(my_biases):
            b_self = my_biases[_id]
        else:
            b_self = 0.0
        
        Y = raw_ratings - global_mean - relevant_biases - b_self
        X = fixed_vecs[target_ids]
        
        XtX = np.dot(X.T, X) + eye * len(target_ids)
        XtY = np.dot(X.T, Y)
        
        try:
            vec = np.linalg.solve(XtX, XtY)
        except:
            XtX += 0.01 * np.eye(rank)
            try: vec = np.linalg.solve(XtX, XtY)
            except: vec = np.zeros(rank, dtype=np.float32)
            
        results.append((_id, vec))
        
    return results

# ==========================================
# Main Training Logic (Adapted from train.py)
# ==========================================
def retrain_model(ratings_df, output_model="model_final.pkl"):
    """
    Retrain the ALS model using the provided ratings DataFrame
    """
    spark = get_spark_session()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    print(f"[{time.strftime('%H:%M:%S')}] Retraining Start. Config: Rank={RANK}, Reg={LAMBDA_REG}")
    print(f"Total ratings to train on: {len(ratings_df)}")
    
    # Convert DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(ratings_df)
    
    raw_rdd = spark_df.select("userId", "movieId", "rating").rdd \
        .filter(lambda r: r.userId is not None and r.movieId is not None) \
        .map(lambda r: (int(r.userId), int(r.movieId), float(r.rating)))
    
    # Split train/val
    train_raw, val_raw = raw_rdd.randomSplit([0.9, 0.1], seed=42)
    train_raw.persist(StorageLevel.MEMORY_AND_DISK)
    val_raw.cache()
    
    # Map IDs
    user_ids = train_raw.map(lambda x: x[0]).distinct().collect()
    item_ids = train_raw.map(lambda x: x[1]).distinct().collect()
    u_map = {uid: i for i, uid in enumerate(user_ids)}
    i_map = {iid: i for i, iid in enumerate(item_ids)}
    u_map_bd = sc.broadcast(u_map)
    i_map_bd = sc.broadcast(i_map)
    
    print(f"Users: {len(user_ids)}, Items: {len(item_ids)}")
    
    # Prepare Train Data
    train_rdd = train_raw.map(lambda x: (
        u_map_bd.value[x[0]], i_map_bd.value[x[1]], x[2]
    )).repartition(PARALLELISM)
    train_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    
    user_data = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list).persist(StorageLevel.MEMORY_AND_DISK)
    item_data = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list).persist(StorageLevel.MEMORY_AND_DISK)
    user_data.count()
    
    # Pre-calculate Biases
    print(f"[{time.strftime('%H:%M:%S')}] Pre-calculating Biases...")
    global_mean = train_raw.map(lambda x: x[2]).mean()
    
    u_bias_dict = train_raw.map(lambda x: (u_map.get(x[0]), (x[2] - global_mean, 1))) \
        .filter(lambda x: x[0] is not None) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
        .mapValues(lambda v: v[0] / (v[1] + BIAS_REG)) \
        .collectAsMap()
        
    u_bias_bd_temp = sc.broadcast(u_bias_dict)
    i_bias_dict = train_raw.map(lambda x: (i_map.get(x[1]), (x[2] - global_mean - u_bias_bd_temp.value.get(u_map.get(x[0]), 0.0), 1))) \
        .filter(lambda x: x[0] is not None) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
        .mapValues(lambda v: v[0] / (v[1] + BIAS_REG)) \
        .collectAsMap()
    u_bias_bd_temp.unpersist()

    num_users, num_items = len(user_ids), len(item_ids)
    B_u = np.zeros(num_users, dtype=np.float32)
    B_i = np.zeros(num_items, dtype=np.float32)
    for u_idx, b in u_bias_dict.items(): B_u[u_idx] = b
    for i_idx, b in i_bias_dict.items(): B_i[i_idx] = b
    
    # Init Vectors
    np.random.seed(42)
    P = np.random.normal(0, 0.1, (num_users, RANK)).astype(np.float32)
    Q = np.random.normal(0, 0.1, (num_items, RANK)).astype(np.float32)
    
    print(f"{'Epoch':<10} | {'Time(s)':<10} | {'Train RMSE':<12}")
    print(f"{'-'*40}")
    
    # Training Loop
    for i in range(ITERATIONS):
        iter_start = time.time()
        
        # Fix Q, update P
        Q_bd = sc.broadcast(Q)
        Bi_bd = sc.broadcast(B_i)
        Bu_bd = sc.broadcast(B_u)
        
        res_u = user_data.mapPartitions(lambda it: solve_batch_residual(it, Q_bd, Bi_bd, Bu_bd, global_mean, RANK, LAMBDA_REG)).collect()
        for u_idx, vec in res_u:
            P[u_idx] = vec
            
        Q_bd.unpersist(); Bi_bd.unpersist(); Bu_bd.unpersist()
        
        # Fix P, update Q
        P_bd = sc.broadcast(P)
        Bu_bd = sc.broadcast(B_u)
        Bi_bd = sc.broadcast(B_i)
        
        res_i = item_data.mapPartitions(lambda it: solve_batch_residual(it, P_bd, Bu_bd, Bi_bd, global_mean, RANK, LAMBDA_REG)).collect()
        for i_idx, vec in res_i:
            Q[i_idx] = vec
            
        P_bd.unpersist(); Bu_bd.unpersist(); Bi_bd.unpersist()
        
        # Monitor
        cP = sc.broadcast(P); cQ = sc.broadcast(Q); cBu = sc.broadcast(B_u); cBi = sc.broadcast(B_i)
        train_rmse = compute_rmse(train_rdd, cP, cQ, cBu, cBi, global_mean)
        cP.unpersist(); cQ.unpersist(); cBu.unpersist(); cBi.unpersist()
        print(f"{i+1:<10} | {time.time()-iter_start:<10.1f} | {train_rmse:<12.5f}")

    # Evaluation
    print(f"[{time.strftime('%H:%M:%S')}] Final Evaluation...")
    val_samples = val_raw.collect()
    errors = []
    
    for u_raw, i_raw, r_true in val_samples:
        if u_raw in u_map and i_raw in i_map:
            u_idx = u_map[u_raw]
            i_idx = i_map[i_raw]
            pred = global_mean + B_u[u_idx] + B_i[i_idx] + np.dot(P[u_idx], Q[i_idx])
            pred = max(0.5, min(5.0, pred))
            err = abs(r_true - pred)
            errors.append(err)
            
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    mae = np.mean(errors)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE:  {mae:.4f}")
    
    # Save Model (Include global_mean this time!)
    with open(output_model, "wb") as f:
        pickle.dump({
            "P": P, 
            "Q": Q, 
            "B_u": B_u, 
            "B_i": B_i, 
            "u_map": u_map, 
            "i_map": i_map,
            "global_mean": global_mean  # Important: Save global_mean
        }, f)
    
    print(f"[{time.strftime('%H:%M:%S')}] Model saved to {output_model}")
    print(f"[{time.strftime('%H:%M:%S')}] Retraining Complete!")
    
    spark.stop()
    return rmse, mae

# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ALS model from MongoDB ratings")
    parser.add_argument('--db-only', action='store_true', 
                        help='Use only MongoDB ratings (ignore CSV data)')
    parser.add_argument('--output', type=str, default='model_final.pkl',
                        help='Output model filename (default: model_final.pkl)')
    parser.add_argument('--csv-path', type=str, default='data/ml-latest-small/ratings.csv',
                        help='Path to original CSV file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Model Retraining Script")
    print("="*60)
    
    # Step 1: Extract and merge data
    ratings_data = merge_ratings_data(csv_path=args.csv_path, use_db_only=args.db_only)
    
    if len(ratings_data) == 0:
        print("ERROR: No ratings data available for training!")
        sys.exit(1)
    
    # Step 2: Retrain model
    retrain_model(ratings_data, output_model=args.output)
