# 🎥 MovieRec 电影推荐系统项目文档

## 📚 1. 核心推荐算法与分布式训练设计 (`train.py`)

本项目采用基于 **矩阵分解 (Matrix Factorization)** 的协同过滤算法，具体使用了 **ALS (Alternating Least Squares, 交替最小二乘法)** 的一种改进变体。代码基于 **Apache PySpark** 框架编写，充分利用了分布式计算能力来加速模型训练。

### 1.1 算法核心逻辑：显式反馈矩阵分解
我们的目标是预测用户 $u$ 对物品 $i$ 的评分 $\hat{r}_{ui}$。传统的矩阵分解预测公式为：

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i^T
$$

其中：
- $\mu$: 全局平均分 (Global Mean)
- $b_u$: 用户偏差 (User Bias)，表示该用户是否倾向于打高分。
- $b_i$: 物品偏差 (Item Bias)，表示该电影是否普遍受欢迎。
- $\mathbf{p}_u$: 用户隐向量 (Latent Factor Vector for User)。
- $\mathbf{q}_i$: 物品隐向量 (Latent Factor Vector for Item)。

### 1.2 针对小数据集的核心优化 (Robustness for Small Data)

由于使用的是 MovieLens Small 数据集 (100k ratings)，标准 ALS 容易过拟合。代码中实现了以下**关键优化**：

#### A. 统计收缩 (Statistical Shrinkage) 预计算 Bias
在迭代开始前，不让模型去"学习" $b_u$ 和 $b_i$（这会增加参数量导致过拟合），而是直接通过统计方法**预计算**并固定下来。
使用了正则化收缩公式：

$$
b_u = \frac{\sum (r_{ui} - \mu)}{N_u + \lambda_{bias}}
$$

其中 $\lambda_{bias}=10$ (`BIAS_REG`)。这意味着如果一个用户评分很少，他的偏差会趋向于0（不敢断定他偏好严苛还是宽松），只有评分数据足够多时，$b_u$ 才会显著。
*代码位置*: `train_with_monitoring` 函数中的 "Pre-calculating Biases" 部分。

#### B. 残差拟合 (Residual Fitting)
在 ALS 迭代求解 $\mathbf{p}_u$ 和 $\mathbf{q}_i$ 时，我们不是由零开始拟合，而是拟合**残差**：
$$ 
\text{Target} = r_{ui} - \mu - b_u - b_i 
$$
模型的任务简化为：让 $\mathbf{p}_u \cdot \mathbf{q}_i^T$ 尽可能接近这个“去除了平均效应和偏差效应”后的残差。这大大降低了隐向量的学习难度，提高了泛化能力。
*代码位置*: `solve_batch_residual` 函数。

### 1.3 分布式计算 (Spark RDD) 实现
代码没有使用 Spark MLlib 的现成库，而是**手写了并行算子**，以便完全控制优化细节。这使得我们能够精确实现上述的统计收缩和残差拟合优化。

#### 分布式训练架构详解

**核心思想**: 利用 Apache Spark 的 RDD (Resilient Distributed Dataset) 弹性分布式数据集，将矩阵分解中的大规模线性代数运算并行化到多个计算节点上。

##### A. 数据分区策略 (Partitioning Strategy)

在 ALS 算法中，更新用户向量 $\mathbf{p}_u$ 时需要该用户的所有评分记录，更新物品向量 $\mathbf{q}_i$ 时需要该物品的所有评分记录。因此我们采用**按 ID 聚合分区**的策略：

```python
# 用户维度分区：将同一用户的所有评分聚合到一起
user_data = train_rdd.map(lambda x: (x[0], (x[1], x[2]))) \
                     .groupByKey() \
                     .mapValues(list)
# 结果: (user_idx, [(item_1, rating_1), (item_2, rating_2), ...])

# 物品维度分区：将同一物品的所有评分聚合到一起  
item_data = train_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
                     .groupByKey() \
                     .mapValues(list)
# 结果: (item_idx, [(user_1, rating_1), (user_2, rating_2), ...])
```

通过 `groupByKey()` 操作，Spark 会自动进行 **Shuffle (数据重分布)**，确保相同 Key 的数据被路由到同一个分区 (Partition)。每个分区可以独立地在一个 Executor (计算节点) 上并行处理。

**分区数量控制**: 通过 `.repartition(PARALLELISM)` 设定为 8 个分区，平衡了并行度和通信开销。

##### B. 广播变量优化 (Broadcast Variables)

ALS 是**交替优化**算法：
- **Step 1**: 固定物品矩阵 Q，更新用户矩阵 P
- **Step 2**: 固定用户矩阵 P，更新物品矩阵 Q

在 Step 1 中，所有 Executor 都需要读取完整的 Q 矩阵。如果不做优化，每个 Task 都会通过网络从 Driver 拉取 Q，造成巨大的网络开销。

**广播机制**:
```python
Q_bd = sc.broadcast(Q)  # 将 Q 矩阵广播到所有节点
# Spark 会使用高效的 P2P 协议 (BitTorrent-like) 分发数据
# 每个 Executor 只需下载一次，然后缓存在本地内存
```

广播变量的优势：
- **网络传输优化**: 从 O(tasks × data_size) 降低到 O(executors × data_size)
- **内存共享**: 同一 Executor 上的多个 Task 共享同一份广播数据
- **只读语义**: 保证数据一致性

类似地，我们也广播了 Bias 数组 (`B_u`, `B_i`) 和全局均值 (`global_mean`)。

##### C. 并行求解 (Parallel Least Squares Solve)

核心计算逻辑封装在 `solve_batch_residual` 函数中，通过 `mapPartitions` 算子实现分区级并行：

```python
res_u = user_data.mapPartitions(
    lambda it: solve_batch_residual(it, Q_bd, Bi_bd, Bu_bd, 
                                    global_mean, RANK, LAMBDA_REG)
).collect()
```

**执行流程**:
1. **分发任务**: Spark 将 `user_data` 的每个分区分配给不同的 Executor
2. **本地计算**: 每个 Executor 独立地对分区内的所有用户求解最小二乘问题
   - 对于用户 $u$，构建目标向量 $Y = r_{ui} - \mu - b_u - b_i$（残差）
   - 构建设计矩阵 $X$ (对应物品的隐向量 $\mathbf{q}_i$)
   - 求解正规方程: $(X^TX + \lambda I) \mathbf{p}_u = X^T Y$
   - 使用 NumPy 的 `linalg.solve` 高效求解（利用 BLAS/LAPACK 加速）
3. **结果收集**: `.collect()` 将所有分区的计算结果汇总回 Driver
4. **更新矩阵**: 在 Driver 端更新 P 矩阵

**为什么用 `mapPartitions` 而不是 `map`**:
- `map` 是逐条记录处理，每条记录都会调用一次函数
- `mapPartitions` 是批量处理整个分区，可以复用计算资源（如预计算的正则化矩阵 $\lambda I$）
- 减少了函数调用开销和序列化开销

##### D. 内存管理与持久化

```python
train_rdd.persist(StorageLevel.MEMORY_AND_DISK)
user_data.persist(StorageLevel.MEMORY_AND_DISK)
```

使用 `.persist()` 将中间结果缓存：
- **MEMORY_AND_DISK**: 优先存内存，内存不足时溢出到磁盘
- 避免重复计算 (Spark 的 RDD 默认是惰性求值)
- 在迭代算法中尤为重要（ALS 需要 20 次迭代）

**显式释放**:
```python
Q_bd.unpersist()  # 每次迭代后释放旧的广播变量
```
防止内存泄漏，因为每次迭代 Q 都会更新，旧的广播变量不再需要。

##### E. 容错机制

Spark RDD 的 **Lineage (血统)** 机制提供了自动容错：
- 每个 RDD 都记录了它的计算依赖关系
- 如果某个分区的数据丢失（节点故障），Spark 可以根据 Lineage 自动重新计算
- 在我们的代码中，通过 `.persist()` 缓存了关键的中间结果，减少了失败后的恢复开销

---

### 1.4 模型增量更新与重训练 (`retrain_from_db.py`)

#### 问题背景
在生产环境中，用户会持续产生新的评分数据，这些数据被实时写入 MongoDB 数据库。但是，推荐引擎使用的是**静态的** `model_final.pkl` 文件，无法自动吸收新数据。这导致：

1. **新用户冷启动**: 新注册用户的 `inner_idx = -1`，只能获得非个性化的热度榜推荐
2. **老用户偏好漂移**: 即使老用户的口味发生变化，推荐结果也不会更新
3. **新电影推荐滞后**: 新增电影无法进入推荐池

#### 解决方案：重训练脚本

`retrain_from_db.py` 是一个独立的模型更新脚本，它能够：

**核心功能**:
1. **从 MongoDB 提取评分数据** - 连接数据库，读取 `ratings` 集合中的所有评分记录
2. **与原始 CSV 数据合并** - 将数据库中的新评分与 MovieLens 原始数据集合并
   - 对于同一 (用户, 电影) 对，**数据库数据优先**（用户可能修改了评分）
   - 使用 Pandas 的 `drop_duplicates(keep='last')` 实现
3. **完整重训练** - 调用与 `train.py` 完全相同的 ALS 训练逻辑
4. **更新模型文件** - 将新模型覆盖写入 `model_final.pkl`

**使用方式**:
```bash
# 标准模式：合并 CSV + MongoDB 数据后重训练
python retrain_from_db.py

# 仅使用数据库数据（忽略原始 CSV）
python retrain_from_db.py --db-only

# 指定输出文件名（用于 A/B 测试）
python retrain_from_db.py --output model_v2.pkl

# 指定 CSV 路径
python retrain_from_db.py --csv-path data/ml-latest-small/ratings.csv
```

#### 数据合并策略

```python
def merge_ratings_data(csv_path, use_db_only=False):
    db_df = extract_ratings_from_db()  # 从 MongoDB 提取
    
    if not use_db_only:
        csv_df = pd.read_csv(csv_path)  # 加载原始数据
        combined_df = pd.concat([csv_df, db_df], ignore_index=True)
        # 去重：对于相同的 (userId, movieId)，保留最后出现的记录
        combined_df = combined_df.drop_duplicates(
            subset=['userId', 'movieId'], keep='last'
        )
    else:
        combined_df = db_df
    
    return combined_df
```

**去重逻辑**: `keep='last'` 确保数据库中的评分会覆盖 CSV 中的旧评分，符合"用户最新行为优先"的原则。

#### 训练一致性保证

`retrain_from_db.py` 与 `train.py` 在训练逻辑上**完全一致**：
- ✅ 相同的超参数 (RANK=20, LAMBDA_REG=0.1, BIAS_REG=10)
- ✅ 相同的预计算 Bias 逻辑（统计收缩）
- ✅ 相同的残差拟合求解器 (`solve_batch_residual`)
- ✅ 相同的分布式计算策略（Spark RDD + 广播变量）

唯一的区别是**数据来源**：
- `train.py`: 直接从 CSV 文件读取
- `retrain_from_db.py`: 从 MongoDB 提取并与 CSV 合并

#### 生产部署建议

**更新频率**:
- **低频更新** (推荐): 每天凌晨定时重训练（使用 cron 或 Task Scheduler）
  ```bash
  # Linux Cron 示例
  0 2 * * * cd /path/to/project && python retrain_from_db.py
  ```
- **中频更新**: 每当新增评分达到一定阈值（如 1000 条）时触发
- **高频更新** (不推荐): 实时重训练会占用大量计算资源

**平滑切换**:
1. 重训练时先输出到 `model_new.pkl`
2. 验证新模型的 RMSE/MAE 指标
3. 如果指标正常，原子性地重命名文件（`mv model_new.pkl model_final.pkl`）
4. Flask 应用检测到文件更新后，重新加载模型（可以添加文件监听或使用信号量）

**新用户处理**:
重训练后，新用户会被分配 `inner_idx`，从"冷启动用户"升级为"个性化推荐用户"。需要更新数据库：
```python
# 训练完成后，更新 MongoDB 中的 inner_idx
for raw_uid, inner_idx in u_map.items():
    db.users.update_one(
        {"userId": raw_uid},
        {"$set": {"inner_idx": inner_idx, "is_new": False}}
    )
```
*(注: 当前脚本尚未实现此步骤，建议在 `init_db.py` 或单独的同步脚本中完成)*

---

## 🛠️ 2. 后端架构与系统运行逻辑

整个系统采用 **Flask (Python)** 作为后端框架，**MongoDB** 作为数据库，前端使用原生 **HTML/JS + TailwindCSS**。

### 2.1 数据库设计 (MongoDB)
数据库 `movie_rec_system` 包含三个核心集合：

1.  **`movies` (电影表)**
    - 存储电影的基础信息：`movieId`, `title`, `genres`。
    - **特有字段**:
        - `inner_idx`: 对应模型矩阵 Q 中的行索引（为了快速查表）。
        - `avg_rating`, `rating_count`: 预计算的统计信息，用于**冷启动推荐**（热度榜）。
    - *数据来源*: `init_db.py` 脚本负责从 `movies.csv` 导入并计算统计值。

2.  **`users` (用户表)**
    - 存储用户信息：`userId`, `username`, `password`。
    - **特有字段**:
        - `inner_idx`: 如果是老用户，对应模型矩阵 P 的行索引；如果是**新注册用户**，该值为 `-1`，标识需要走冷启动逻辑。
        - `is_new`: 标识用户类型。

3.  **`ratings` (评分表)**
    - 存储所有评分记录：`userId`, `movieId`, `rating`, `timestamp`。
    - 用于记录用户看了什么，以及后续增量训练。

### 2.2 推荐引擎 (`MovieRecommender` in `recommender.py`)
这是一个独立的类，负责加载训练好的 `model_final.pkl` 文件。

*   **加载逻辑**: 读取 Pickle 文件中的 P, Q 矩阵和预计算的 Bias。
*   **个性化推荐 (`get_personalized_recommendations`)**:
    对于老用户（`inner_idx != -1`），直接计算：
    $$ \text{Score}_i = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{Q}^T $$
    利用矩阵乘法 `np.dot` 一次性计算该用户对所有电影的预测分，排序后取 Top-N，并过滤掉已看过的电影。
*   **相似推荐 (`get_similar_movies`)**:
    基于 Item-Based 协同过滤。计算电影隐向量 $\mathbf{q}_i$ 之间的**余弦相似度**，用于详情页的 "Similar Movies" 功能。

### 2.3 业务流程控制 (`app.py`)

#### A. 用户鉴权 (Auth)
*   **注册**: 新用户注册时，被分配一个新的 `userId` (1000+)，且 `inner_idx` 设为 -1。
*   **登录**: 使用 Session 保持状态。

#### B. 核心推荐路由 (`/api/recommend`)
这是系统的核心分流器：
1.  **检查用户状态**:
    从 Session 获取 `user_id`，查询数据库。
2.  **分支一：老用户 (Personalized)**
    - 如果 `user.inner_idx != -1`。
    - 调用 `engine.get_personalized_recommendations()`。
    - 返回基于矩阵分解的高分预测电影。
3.  **分支二：新用户 (Cold Start)**
    - 如果 `user.inner_idx == -1`（新注册用户，没有隐向量）。
    - **降级策略**: 按照非个性化的“热度榜”推荐。
    - 查询 MongoDB，按 `avg_rating` 降序 (`sort([("avg_rating", -1), ("rating_count", -1)])`) 取前 10 名。

#### C. 实时评分反馈 (`/api/rate`)
当用户在前端打分时：
1.  插入/更新 MongoDB 的 `ratings` 表。
2.  **实时更新统计**: 立即重新计算该电影的 `avg_rating` 并更新 `movies` 表。这保证了冷启动推荐（热度榜）是实时的。
    *(注: 用户的隐向量 P 不会实时更新，需要定期重新运行 `train.py` 才能吸纳新数据)*

### 2.4 运行步骤

0.  **环境准备**:
    *   **安装依赖**:
        ```bash
        pip install -r requirements.txt
        ```
    *   **配置 MongoDB**:
        请确保本地已安装 MongoDB 并且服务正在运行（默认端口 `27017`）。无需手动创建数据库，后续的初始化脚本会自动处理。
    *   **关于数据集 (可选)**:
        `train.py` 会自动下载数据集。如果你不想运行训练脚本（通过其他方式获取了 `model_final.pkl`），需要手动下载数据集以供数据库初始化使用：
        
        **方法 A：手动下载**
        1. 下载 [MovieLens Small 数据集](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)。
        2. 解压至项目根目录的 `data/` 文件夹下，确保路径结构为 `data/ml-latest-small/ratings.csv`。

        **方法 B：终端命令下载 (Windows PowerShell)**
        ```powershell
        mkdir data
        Invoke-WebRequest -Uri "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip" -OutFile "data/ml-latest-small.zip"
        Expand-Archive -Path "data/ml-latest-small.zip" -DestinationPath "data/"
        ```

1.  **训练模型**:
    ```bash
    python train.py
    # 生成 model_final.pkl (包含矩阵 P, Q, Biases)
    ```

2.  **初始化数据库**:
    ```bash
    python init_db.py
    # 读取 csv 和 pkl，将数据导入 MongoDB
    # 计算初始的电影热度统计数据
    ```

3.  **启动服务器**:
    ```bash
    python app.py
    # 访问 http://127.0.0.1:5000
    ```

4.  **模型更新** (生产环境定期执行):
    ```bash
    # 用户产生新评分后，定期重训练模型以更新推荐
    python retrain_from_db.py
    # 合并 MongoDB 和 CSV 数据，重新训练并覆盖 model_final.pkl
    
    # 重启 Flask 应用以加载新模型
    # (或实现热加载机制)
    ```
