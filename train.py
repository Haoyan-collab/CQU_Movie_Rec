import os
import time
import sys
import pickle
import zipfile
import requests
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkConf, StorageLevel

# ==========================================
# 1. 配置与超参数
# ==========================================
DRIVER_MEMORY = "2g"
EXECUTOR_MEMORY = "4g"
PARALLELISM = 8

# === 核心优化: 针对小数据集的黄金参数 ===
RANK = 20           # [优化] 降低秩防止过拟合
ITERATIONS = 20     # [保持]
LAMBDA_REG = 0.1    # [保持]
BIAS_REG = 10       # [优化] 统计Bias时的收缩系数 (关键！)

def get_spark_session():
    conf = SparkConf() \
        .setAppName("MovieLens_ALS_Final_Robust") \
        .setMaster("local[*]") \
        .set("spark.driver.memory", DRIVER_MEMORY) \
        .set("spark.executor.memory", EXECUTOR_MEMORY) \
        .set("spark.default.parallelism", str(PARALLELISM)) \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    return SparkSession.builder.config(conf=conf).getOrCreate()

# ==========================================
# 2. 数据准备 (完全保留)
# ==========================================
def prepare_small_data(data_dir="data"):
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    extract_path = os.path.join(data_dir, "ml-latest-small")
    
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    if not os.path.exists(os.path.join(extract_path, "ratings.csv")):
        try:
            r = requests.get(url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): 
                    if chunk: f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
        except Exception as e:
            pass 
            
    base_dir = os.path.join(data_dir, "ml-latest-small")
    if os.path.exists(os.path.join(base_dir, "ratings.csv")):
        return os.path.join(base_dir, "ratings.csv")
    return os.path.join(data_dir, "ratings.csv")

# ==========================================
# 3. 辅助: RMSE 计算 (完全保留)
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
# 4. 核心 Solver (替换为：残差最小二乘法)
# ==========================================
# 彻底修复了 AttributeError: 'numpy.ndarray' object has no attribute 'get'
def solve_batch_residual(iterator, fixed_vecs_bd, fixed_biases_bd, my_biases_bd, global_mean, rank, lambda_reg):
    fixed_vecs = fixed_vecs_bd.value
    fixed_biases = fixed_biases_bd.value
    my_biases = my_biases_bd.value # 这是一个 Array
    
    eye = np.eye(rank) * lambda_reg
    results = []
    
    for _id, ratings in iterator:
        if not ratings: continue
        
        target_ids = [r[0] for r in ratings]
        raw_ratings = np.array([r[1] for r in ratings], dtype=np.float32)
        
        # 对方的 Bias (Item的或User的)
        relevant_biases = fixed_biases[target_ids]
        
        # 自己的 Bias (直接用下标取，不用 get)
        # _id 已经是 map 后的 int 索引
        if 0 <= _id < len(my_biases):
            b_self = my_biases[_id]
        else:
            b_self = 0.0
        
        # === 核心逻辑: 拟合残差 ===
        # 目标值 = 原始评分 - 全局均值 - 对方Bias - 自己Bias
        # 这样向量 V 就只需要去学 "剩下的个性化偏好"
        Y = raw_ratings - global_mean - relevant_biases - b_self
        
        X = fixed_vecs[target_ids]
        
        XtX = np.dot(X.T, X) + eye * len(target_ids)
        XtY = np.dot(X.T, Y)
        
        try:
            vec = np.linalg.solve(XtX, XtY)
        except:
            # 处理矩阵奇异
            XtX += 0.01 * np.eye(rank)
            try: vec = np.linalg.solve(XtX, XtY)
            except: vec = np.zeros(rank, dtype=np.float32)
            
        results.append((_id, vec))
        
    return results

# ==========================================
# 5. 主流程
# ==========================================
def train_with_monitoring(spark, ratings_path):
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print(f"[{time.strftime('%H:%M:%S')}] Training Start. Config: Rank={RANK}, Reg={LAMBDA_REG}")
    
    # Load
    df = spark.read.csv(ratings_path, header=True, inferSchema=True)
    raw_rdd = df.select("userId", "movieId", "rating").rdd \
        .filter(lambda r: r.userId is not None and r.movieId is not None) \
        .map(lambda r: (int(r.userId), int(r.movieId), float(r.rating)))
    
    # [优化] 恢复成 9:1 或 8:2，98:2 对于小数据集太容易过拟合且验证不准
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
    
    # Prepare Train Data
    train_rdd = train_raw.map(lambda x: (
        u_map_bd.value[x[0]], i_map_bd.value[x[1]], x[2]
    )).repartition(PARALLELISM)
    train_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    
    user_data = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list).persist(StorageLevel.MEMORY_AND_DISK)
    item_data = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list).persist(StorageLevel.MEMORY_AND_DISK)
    user_data.count() # Trigger
    
    # === [关键优化] 预计算 Statistical Biases ===
    # 这一步替代了原来的 B_u, B_i 初始化为0
    print(f"[{time.strftime('%H:%M:%S')}] Pre-calculating Biases (Statistical Shrinkage)...")
    global_mean = train_raw.map(lambda x: x[2]).mean()
    
    # 1. 计算 User Bias
    u_bias_dict = train_raw.map(lambda x: (u_map.get(x[0]), (x[2] - global_mean, 1))) \
        .filter(lambda x: x[0] is not None) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
        .mapValues(lambda v: v[0] / (v[1] + BIAS_REG)) \
        .collectAsMap()
        
    # 2. 计算 Item Bias
    u_bias_bd_temp = sc.broadcast(u_bias_dict)
    i_bias_dict = train_raw.map(lambda x: (i_map.get(x[1]), (x[2] - global_mean - u_bias_bd_temp.value.get(u_map.get(x[0]), 0.0), 1))) \
        .filter(lambda x: x[0] is not None) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
        .mapValues(lambda v: v[0] / (v[1] + BIAS_REG)) \
        .collectAsMap()
    u_bias_bd_temp.unpersist()

    # 填入 Numpy Array (保持数据结构，方便后续 SaveModel)
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
    
    # Loop
    for i in range(ITERATIONS):
        iter_start = time.time()
        
        # Fix Q, update P
        Q_bd = sc.broadcast(Q)
        Bi_bd = sc.broadcast(B_i) # 固定传入 B_i
        Bu_bd = sc.broadcast(B_u) # 固定传入 B_u
        
        # 调用新的 Solver
        res_u = user_data.mapPartitions(lambda it: solve_batch_residual(it, Q_bd, Bi_bd, Bu_bd, global_mean, RANK, LAMBDA_REG)).collect()
        for u_idx, vec in res_u:
            P[u_idx] = vec
            # B_u 不更新，因为是预计算好的
            
        Q_bd.unpersist(); Bi_bd.unpersist(); Bu_bd.unpersist()
        
        # Fix P, update Q
        P_bd = sc.broadcast(P)
        Bu_bd = sc.broadcast(B_u)
        Bi_bd = sc.broadcast(B_i)
        
        res_i = item_data.mapPartitions(lambda it: solve_batch_residual(it, P_bd, Bu_bd, Bi_bd, global_mean, RANK, LAMBDA_REG)).collect()
        for i_idx, vec in res_i:
            Q[i_idx] = vec
            # B_i 不更新
            
        P_bd.unpersist(); Bu_bd.unpersist(); Bi_bd.unpersist()
        
        # Monitor (保持原样)
        cP = sc.broadcast(P); cQ = sc.broadcast(Q); cBu = sc.broadcast(B_u); cBi = sc.broadcast(B_i)
        train_rmse = compute_rmse(train_rdd, cP, cQ, cBu, cBi, global_mean)
        cP.unpersist(); cQ.unpersist(); cBu.unpersist(); cBi.unpersist()
        print(f"{i+1:<10} | {time.time()-iter_start:<10.1f} | {train_rmse:<12.5f}")

    # Evaluation (完全保留)
    print(f"[{time.strftime('%H:%M:%S')}] Final Evaluation...")
    val_samples = val_raw.collect()
    errors = []
    results_csv = []
    
    for u_raw, i_raw, r_true in val_samples:
        if u_raw in u_map and i_raw in i_map:
            u_idx = u_map[u_raw]
            i_idx = i_map[i_raw]
            # 这里预测逻辑依然兼容：Mean + Bu + Bi + Dot
            pred = global_mean + B_u[u_idx] + B_i[i_idx] + np.dot(P[u_idx], Q[i_idx])
            pred = max(0.5, min(5.0, pred))
            err = abs(r_true - pred)
            errors.append(err)
            results_csv.append(f"{u_raw},{i_raw},{r_true},{pred:.4f},{err:.4f}")
            
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    mae = np.mean(errors)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE:  {mae:.4f}")
    
    # Save Model & CSV (你要的都在这)
    with open("model_final.pkl", "wb") as f:
        # B_u 和 B_i 已经被填入了预计算的值，保存下来的模型可以直接用
        pickle.dump({"P":P, "Q":Q, "B_u":B_u, "B_i":B_i, "u_map":u_map, "i_map":i_map}, f)
        
    with open("predictions.csv", "w") as f:
        f.write("userId,movieId,actual,predicted,error\n")
        for line in results_csv: f.write(line + "\n")
        
        f.write("\n") 
        f.write("=== Summary Statistics ===\n")
        f.write(f"Mean Absolute Error (MAE),{mae:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE),{rmse:.4f}\n")
        
    print("Predictions saved to predictions.csv with summary stats.")

if __name__ == "__main__":
    data_path = prepare_small_data()
    spark = get_spark_session()
    train_with_monitoring(spark, data_path)
    spark.stop()