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
在 ALS 迭代求解 $\mathbf{p}_u$ 和 $\mathbf{q}_i$ 时，我们不是由零开始拟合，而是拟合**残差**。
$$ \text{Target} = r_{ui} - \mu - b_u - b_i $$
模型的任务简化为：让 $\mathbf{p}_u \cdot \mathbf{q}_i^T$ 尽可能接近这个“去除了平均效应和偏差效应”后的残差。这大大降低了隐向量的学习难度，提高了泛化能力。
*代码位置*: `solve_batch_residual` 函数。

### 1.3 分布式计算 (Spark RDD) 实现
代码没有使用 Spark MLlib 的现成库，而是**手写了并行算子**，以便完全控制优化细节。

1.  **数据分区 (Partitioning)**:
    - `user_data` 和 `item_data` 分别按照 UserID 和 MovieID 进行 `groupByKey`。
    - 这样可以将同一个用户的所有评分数据汇聚到同一个计算节点上，方便一次性算出该用户的向量 $\mathbf{p}_u$。

2.  **广播变量 (Broadcast Variables)**:
    - ALS 是交替优化的：固定 Q 算 P，然后固定 P 算 Q。
    - 在算 P 时，矩阵 Q 是固定的。代码将 Q 矩阵通过 `sc.broadcast(Q)` 广播到所有工作节点，大大减少了网络传输开销。

3.  **并行求解 (Parallel Solve)**:
    - 使用 `mapPartitions` 在每个分区内并行运行 `solve_batch_residual`。
    - 内部利用 `numpy.linalg.solve` 求解最小二乘问题 ($X^TX + \lambda I)^{-1} X^TY$。

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
