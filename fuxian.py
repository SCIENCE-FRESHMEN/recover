import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import random
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')


class PMF:
    """自定义概率矩阵分解(PMF)实现"""

    def __init__(self, n_factors=10, n_epochs=20, lr=0.01, reg=0.1, seed=None):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def fit(self, ratings_matrix):
        """训练PMF模型"""
        n_users, n_items = ratings_matrix.shape
        mask = ratings_matrix > 0

        # 初始化嵌入
        self.user_embeddings = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_embeddings = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 中心化评分
        user_means = np.zeros(n_users)
        for i in range(n_users):
            user_ratings = ratings_matrix[i, mask[i]]
            if len(user_ratings) > 0:
                user_means[i] = np.mean(user_ratings)
                ratings_matrix[i, mask[i]] -= user_means[i]

        self.user_means = user_means

        # 训练
        for epoch in range(self.n_epochs):
            for i in range(n_users):
                for j in range(n_items):
                    if mask[i, j]:
                        pred = np.dot(self.user_embeddings[i], self.item_embeddings[j])
                        error = ratings_matrix[i, j] - pred

                        # 更新嵌入
                        self.user_embeddings[i] += self.lr * (
                                    error * self.item_embeddings[j] - self.reg * self.user_embeddings[i])
                        self.item_embeddings[j] += self.lr * (
                                    error * self.user_embeddings[i] - self.reg * self.item_embeddings[j])

        return self

    def predict(self, user_idx, item_idx):
        """预测评分"""
        return np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx]) + self.user_means[user_idx]


class NMFWrapper:
    """NMF封装器，与PMF接口保持一致"""

    def __init__(self, n_factors=10, max_iter=200, reg=0.1, seed=None):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.reg = reg
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def fit(self, ratings_matrix):
        """训练NMF模型"""
        # 创建NMF实例
        model = NMF(n_components=self.n_factors,
                    max_iter=self.max_iter,
                    init='random',
                    random_state=self.seed if self.seed is not None else None)

        # 训练模型
        W = model.fit_transform(ratings_matrix)
        H = model.components_

        self.user_embeddings = W
        self.item_embeddings = H.T
        self.user_means = np.zeros(ratings_matrix.shape[0])  # NMF不中心化

        return self

    def predict(self, user_idx, item_idx):
        """预测评分"""
        return np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])


class ExposureGame:
    """
    曝光博弈模型的完整实现
    """

    def __init__(self, user_embeddings, item_dim, temperature=0.1, non_negative=False,
                 algorithm_type='softmax', demand_distribution=None, seed=None):
        """
        初始化曝光博弈
        """
        if seed is not None:
            np.random.seed(seed)

        self.user_embeddings = user_embeddings
        self.num_users = user_embeddings.shape[0]
        self.item_dim = item_dim
        self.temperature = temperature
        self.non_negative = non_negative
        self.algorithm_type = algorithm_type.lower()

        # 创建需求分布
        if demand_distribution is None:
            self.demand_distribution = np.ones(self.num_users) / self.num_users
        else:
            self.demand_distribution = demand_distribution

        # 验证输入
        assert self.algorithm_type in ['softmax', 'hardmax'], "algorithm_type必须是'softmax'或'hardmax'"

    def _compute_similarities(self, producer_strategies, user_embeddings=None):
        """计算生产者策略与用户嵌入的相似度"""
        if user_embeddings is None:
            user_embeddings = self.user_embeddings

        # 确保策略在单位球面上
        normalized_strategies = producer_strategies.copy()
        norms = np.linalg.norm(normalized_strategies, axis=1, keepdims=True)
        normalized_strategies = normalized_strategies / (norms + 1e-8)

        if self.non_negative:
            normalized_strategies = np.maximum(normalized_strategies, 0)
            normalized_strategies = normalized_strategies / (
                        np.linalg.norm(normalized_strategies, axis=1, keepdims=True) + 1e-8)

        return np.dot(user_embeddings, normalized_strategies.T)

    def utility(self, producer_strategies, producer_idx=None):
        """
        计算生产者效用（曝光率）
        """
        num_producers = producer_strategies.shape[0]
        similarities = self._compute_similarities(producer_strategies)

        # 计算曝光概率
        if self.temperature <= 0 or self.algorithm_type == 'hardmax':
            # Hardmax情况
            probs = np.zeros_like(similarities)
            max_indices = np.argmax(similarities, axis=1)
            probs[np.arange(len(max_indices)), max_indices] = 1.0
        else:
            # Softmax情况
            exp_sim = np.exp(similarities / self.temperature)
            probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        # 计算期望效用
        utilities = np.zeros(num_producers)
        for i in range(num_producers):
            utilities[i] = np.sum(self.demand_distribution * probs[:, i])

        if producer_idx is not None:
            return utilities[producer_idx]
        return utilities

    def gradient(self, producer_strategies, producer_idx):
        """计算效用函数的梯度"""
        if self.temperature <= 0 or self.algorithm_type == 'hardmax':
            raise ValueError("Hardmax情况下梯度不连续，无法直接计算")

        eps = 1e-6
        grad = np.zeros(self.item_dim)
        base_utility = self.utility(producer_strategies, producer_idx)

        # 有限差分法计算梯度
        for d in range(self.item_dim):
            perturbed_strategies = producer_strategies.copy()
            perturbed_strategies[producer_idx, d] += eps

            # 应用约束
            if self.non_negative:
                perturbed_strategies[producer_idx] = np.maximum(perturbed_strategies[producer_idx], 0)

            # 归一化到单位球面
            norm = np.linalg.norm(perturbed_strategies[producer_idx])
            if norm > 0:
                perturbed_strategies[producer_idx] = perturbed_strategies[producer_idx] / norm

            new_utility = self.utility(perturbed_strategies, producer_idx)
            grad[d] = (new_utility - base_utility) / eps

        return grad

    def riemannian_gradient(self, producer_strategies, producer_idx):
        """计算黎曼梯度"""
        if self.temperature <= 0 or self.algorithm_type == 'hardmax':
            # Hardmax情况使用近似方法
            return self.gradient(producer_strategies, producer_idx)

        grad = self.gradient(producer_strategies, producer_idx)
        s = producer_strategies[producer_idx]
        s_normalized = s / np.linalg.norm(s)
        projection = np.eye(self.item_dim) - np.outer(s_normalized, s_normalized)
        return projection @ grad

    def find_equilibrium(self, num_producers, max_iter=1000, lr=0.1, tol=1e-6,
                         initialization='random', seed=None):
        """
        寻找ε-局部纳什均衡
        """
        if seed is not None:
            np.random.seed(seed)

        # 初始化生产者策略
        if initialization == 'random':
            if self.non_negative:
                strategies = np.abs(np.random.randn(num_producers, self.item_dim))
            else:
                strategies = np.random.randn(num_producers, self.item_dim)
        elif initialization == 'uniform':
            # 在单位球面上均匀分布
            strategies = np.random.randn(num_producers, self.item_dim)
        else:
            raise ValueError(f"不支持的初始化方法: {initialization}")

        # 归一化到单位球面
        for i in range(num_producers):
            if self.non_negative:
                strategies[i] = np.maximum(strategies[i], 0)
            norm = np.linalg.norm(strategies[i])
            if norm > 0:
                strategies[i] = strategies[i] / norm

        # 优化过程
        utility_history = []
        strategy_history = [strategies.copy()]

        for iteration in tqdm(range(max_iter),
                              desc=f"寻找均衡 (τ={self.temperature:.3f}, {'NMF' if self.non_negative else 'PMF'})",
                              leave=False):
            old_strategies = strategies.copy()
            current_utilities = self.utility(strategies)
            utility_history.append(np.mean(current_utilities))

            # 每个生产者根据梯度更新策略
            for i in range(num_producers):
                if self.temperature > 0:
                    grad = self.riemannian_gradient(strategies, i)
                    strategies[i] += lr * grad
                else:  # Hardmax情况使用随机梯度上升
                    # 随机采样小扰动
                    perturbation = np.random.randn(self.item_dim) * lr
                    test_strategies = strategies.copy()
                    test_strategies[i] += perturbation
                    norm = np.linalg.norm(test_strategies[i])
                    if norm > 0:
                        test_strategies[i] = test_strategies[i] / norm

                    # 比较效用
                    if self.utility(test_strategies, i) > current_utilities[i]:
                        strategies[i] = test_strategies[i]

                # 应用约束
                if self.non_negative:
                    strategies[i] = np.maximum(strategies[i], 0)

                # 归一化到单位球面
                norm = np.linalg.norm(strategies[i])
                if norm > 0:
                    strategies[i] = strategies[i] / norm

            strategy_history.append(strategies.copy())

            # 检查收敛
            if np.linalg.norm(strategies - old_strategies) < tol:
                break

        return {
            'strategies': strategies,
            'utility_history': utility_history,
            'strategy_history': strategy_history,
            'final_utilities': self.utility(strategies)
        }

    def analyze_clustering(self, strategies, threshold=1e-3):
        """
        分析生产者策略的聚类情况
        """
        num_clusters = 0
        visited = np.zeros(len(strategies), dtype=bool)
        cluster_assignments = np.zeros(len(strategies), dtype=int)
        cluster_id = 0

        for i in range(len(strategies)):
            if visited[i]:
                continue

            num_clusters += 1
            cluster_assignments[i] = cluster_id
            visited[i] = True

            # 找到与当前策略相似的所有策略
            for j in range(i + 1, len(strategies)):
                if not visited[j] and np.linalg.norm(strategies[i] - strategies[j]) < threshold:
                    visited[j] = True
                    cluster_assignments[j] = cluster_id

            cluster_id += 1

        return num_clusters / len(strategies), cluster_assignments


class PreDeploymentAudit:
    """
    预部署审计框架
    """

    def __init__(self, ratings_matrix, user_ids=None, item_ids=None, user_gender=None,
                 creator_gender=None, algorithm='pmf'):
        """
        初始化预部署审计
        """
        self.ratings_matrix = ratings_matrix
        self.user_ids = user_ids if user_ids is not None else np.arange(ratings_matrix.shape[0])
        self.item_ids = item_ids if item_ids is not None else np.arange(ratings_matrix.shape[1])
        self.user_gender = user_gender
        self.creator_gender = creator_gender
        self.algorithm = algorithm.lower()

        # 验证输入
        assert self.algorithm in ['pmf', 'nmf'], "algorithm必须是'pmf'或'nmf'"

    def train_recommender(self, embedding_dim=10, seed=None):
        """训练推荐模型"""
        if self.algorithm == 'pmf':
            model = PMF(n_factors=embedding_dim, n_epochs=20, lr=0.01, reg=0.1, seed=seed)
        else:  # nmf
            model = NMFWrapper(n_factors=embedding_dim, max_iter=200, reg=0.1, seed=seed)

        model.fit(self.ratings_matrix.copy())
        return model

    def run_audit(self, dimensions=[3, 50], temperatures=[0.01, 0.1, 1.0],
                  num_producers=10, max_iter=1000, lr=0.1, seed=42):
        """
        运行完整的预部署审计
        """
        results = {}

        for dim in dimensions:
            results[dim] = {}
            print(f"\n处理嵌入维度: {dim}")

            # 训练推荐模型
            model = self.train_recommender(embedding_dim=dim, seed=seed)
            user_embeddings = model.user_embeddings.copy()

            for tau in temperatures:
                print(f"  温度 τ={tau}")
                game = ExposureGame(
                    user_embeddings=user_embeddings,
                    item_dim=dim,
                    temperature=tau,
                    non_negative=(self.algorithm == 'nmf'),
                    algorithm_type='softmax',
                    seed=seed
                )

                # 寻找均衡
                equilibrium_result = game.find_equilibrium(
                    num_producers=num_producers,
                    max_iter=max_iter,
                    lr=lr,
                    seed=seed
                )

                # 分析结果
                strategies = equilibrium_result['strategies']
                clustering_ratio, clusters = game.analyze_clustering(strategies)

                results[dim][tau] = {
                    'equilibrium': equilibrium_result,
                    'clustering_ratio': clustering_ratio,
                    'clusters': clusters,
                    'user_embeddings': user_embeddings
                }

        return results

    def visualize_results(self, audit_results, dataset_name="Dataset"):
        """
        可视化审计结果
        """
        plt.figure(figsize=(15, 10))

        # 1. 聚类分析
        plt.subplot(2, 2, 1)

        # 绘制不同维度和温度的结果
        for dim in sorted(audit_results.keys()):
            taus = sorted(audit_results[dim].keys())
            clustering_ratios = [audit_results[dim][tau]['clustering_ratio'] for tau in taus]

            plt.plot(taus, clustering_ratios, 'o-',
                     label=f'dim={dim}',
                     linewidth=2, markersize=8)

        plt.xscale('log')
        plt.xlabel('温度 (τ)', fontsize=12)
        plt.ylabel('聚类数/生产者数', fontsize=12)
        plt.title(f'{dataset_name}: 战略生产者聚类与探索水平', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 2. 策略可视化（2D情况）
        plt.subplot(2, 2, 2)
        found_2d = False

        for dim in sorted(audit_results.keys()):
            if dim == 2:
                found_2d = True
                tau = sorted(audit_results[dim].keys())[0]  # 取第一个温度
                strategies = audit_results[dim][tau]['equilibrium']['strategies']
                user_embeddings = audit_results[dim][tau]['user_embeddings']

                # 可视化
                plt.scatter(strategies[:, 0], strategies[:, 1],
                            c='red', s=100, alpha=0.8, label='生产者策略')
                plt.scatter(user_embeddings[:, 0], user_embeddings[:, 1],
                            c='blue', s=50, alpha=0.6, label='用户嵌入')

                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                plt.xlabel('维度 1', fontsize=12)
                plt.ylabel('维度 2', fontsize=12)
                plt.title(f'{dataset_name}: 2D策略可视化', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                break

        if not found_2d:
            plt.text(0.5, 0.5, '无2D结果可用', ha='center', va='center', fontsize=14)
            plt.axis('off')

        # 3. 效用历史
        plt.subplot(2, 2, 3)
        for dim in sorted(audit_results.keys()):
            tau = sorted(audit_results[dim].keys())[0]  # 取第一个温度
            utility_history = audit_results[dim][tau]['equilibrium']['utility_history']
            plt.plot(utility_history, label=f'dim={dim}')

        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('平均效用', fontsize=12)
        plt.title(f'{dataset_name}: 效用收敛历史', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 4. 最终策略分布
        plt.subplot(2, 2, 4)
        for dim in sorted(audit_results.keys()):
            tau = sorted(audit_results[dim].keys())[-1]  # 取最后一个温度
            strategies = audit_results[dim][tau]['equilibrium']['strategies']

            # 计算策略之间的平均距离
            avg_distances = []
            for i in range(len(strategies)):
                distances = np.linalg.norm(strategies[i] - strategies, axis=1)
                avg_distances.append(np.mean(distances[distances > 0]))

            plt.hist(avg_distances, bins=10, alpha=0.5, label=f'dim={dim}')

        plt.xlabel('策略之间的平均距离', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.title(f'{dataset_name}: 策略分布', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{dataset_name.lower()}_audit_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        return plt.gcf()


def load_movielens_data():
    """加载并预处理MovieLens数据集"""
    print("正在生成模拟的MovieLens数据...")

    # 模拟MovieLens-100K数据
    np.random.seed(42)
    num_users = 943
    num_items = 1682
    density = 0.06  # 6%的密度，接近MovieLens-100K

    # 生成稀疏评分矩阵
    ratings_matrix = np.zeros((num_users, num_items))
    num_ratings = int(num_users * num_items * density)

    # 随机选择用户-物品对
    user_indices = np.random.choice(num_users, num_ratings)
    item_indices = np.random.choice(num_items, num_ratings)

    # 生成1-5的评分
    ratings = np.random.randint(1, 6, num_ratings)

    # 填充矩阵
    for i in range(num_ratings):
        ratings_matrix[user_indices[i], item_indices[i]] = ratings[i]

    # 模拟性别数据（29%女性，71%男性）
    user_gender = np.random.choice([0, 1], size=num_users, p=[0.29, 0.71])

    return ratings_matrix, user_gender


def load_lastfm_data():
    """加载并预处理LastFM数据集（模拟）"""
    print("正在生成模拟的LastFM数据...")

    # 模拟LastFM-360K数据
    np.random.seed(42)
    num_users = 1000
    num_items = 5000
    density = 0.01  # 1%的密度，接近LastFM-360K

    # 生成稀疏评分矩阵
    ratings_matrix = np.zeros((num_users, num_items))
    num_ratings = int(num_users * num_items * density)

    # 随机选择用户-物品对
    user_indices = np.random.choice(num_users, num_ratings)
    item_indices = np.random.choice(num_items, num_ratings)

    # 生成1-5的评分（LastFM是隐式反馈，这里模拟显式评分）
    ratings = np.random.randint(1, 6, num_ratings)

    # 填充矩阵
    for i in range(num_ratings):
        ratings_matrix[user_indices[i], item_indices[i]] = ratings[i]

    # 模拟性别数据（40%女性，60%男性）
    user_gender = np.random.choice([0, 1], size=num_users, p=[0.4, 0.6])

    return ratings_matrix, user_gender


def run_complete_audit():
    """运行完整的审计流程"""
    print("=== 开始完整审计复现 ===")

    # 1. 加载MovieLens数据
    print("\n1. 生成MovieLens数据...")
    movielens_matrix, movielens_gender = load_movielens_data()
    print(f"   MovieLens数据形状: {movielens_matrix.shape}")

    # 2. 运行PMF审计
    print("\n2. 运行PMF审计...")
    pmf_audit = PreDeploymentAudit(
        ratings_matrix=movielens_matrix,
        user_gender=movielens_gender,
        algorithm='pmf'
    )

    pmf_results = pmf_audit.run_audit(
        dimensions=[3, 50],
        temperatures=[0.01, 0.1, 1.0],
        num_producers=10,
        max_iter=500,
        lr=0.1,
        seed=42
    )

    # 3. 运行NMF审计
    print("\n3. 运行NMF审计...")
    nmf_audit = PreDeploymentAudit(
        ratings_matrix=movielens_matrix,
        user_gender=movielens_gender,
        algorithm='nmf'
    )

    nmf_results = nmf_audit.run_audit(
        dimensions=[3, 50],
        temperatures=[0.01, 0.1, 1.0],
        num_producers=10,
        max_iter=500,
        lr=0.1,
        seed=42
    )

    # 4. 合并结果用于可视化
    combined_results = {}
    for dim in pmf_results:
        combined_results[dim] = {}
        for tau in pmf_results[dim]:
            combined_results[dim][tau] = {
                'pmf': pmf_results[dim][tau],
                'nmf': nmf_results[dim][tau]
            }

    # 5. 创建比较可视化
    plt.figure(figsize=(12, 8))

    # 绘制不同算法、维度和温度的聚类结果
    for dim in [3, 50]:
        for algo in ['pmf', 'nmf']:
            taus = sorted(combined_results[dim].keys())
            clustering_ratios = []

            for tau in taus:
                if algo == 'pmf':
                    clustering_ratio = combined_results[dim][tau]['pmf']['clustering_ratio']
                else:
                    clustering_ratio = combined_results[dim][tau]['nmf']['clustering_ratio']
                clustering_ratios.append(clustering_ratio)

            linestyle = '-' if algo == 'pmf' else '--'
            marker = 'o' if dim == 3 else 's'
            plt.plot(taus, clustering_ratios, linestyle=linestyle, marker=marker,
                     label=f'{algo.upper()}, dim={dim}',
                     linewidth=2, markersize=8)

    plt.xscale('log')
    plt.xlabel('温度 (τ)', fontsize=12)
    plt.ylabel('聚类数/生产者数', fontsize=12)
    plt.title('MovieLens: PMF vs NMF 战略生产者聚类', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('movie_lens_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 保存结果
    print("\n6. 保存结果...")
    import pickle
    with open('pmf_audit_results.pkl', 'wb') as f:
        pickle.dump(pmf_results, f)
    with open('nmf_audit_results.pkl', 'wb') as f:
        pickle.dump(nmf_results, f)

    print("\n=== 审计完成 ===")
    print("结果已保存到 'pmf_audit_results.pkl' 和 'nmf_audit_results.pkl'")
    print("可视化已保存到 'movie_lens_comparison.png'")

    return pmf_results, nmf_results


if __name__ == "__main__":
    # 运行完整审计
    pmf_results, nmf_results = run_complete_audit()

    # 额外：演示一个小规模案例
    print("\n=== 演示小规模案例 ===")
    np.random.seed(42)

    # 创建一个小的用户嵌入矩阵
    small_user_embeddings = np.array([
        [1, 0],  # 用户1
        [0.8, 0.6],  # 用户2
        [0, 1],  # 用户3
        [-0.6, 0.8],  # 用户4
        [-1, 0]  # 用户5
    ])

    # 创建曝光博弈
    small_game = ExposureGame(
        user_embeddings=small_user_embeddings,
        item_dim=2,
        temperature=0.1,
        non_negative=False,
        seed=42
    )

    # 寻找均衡
    small_equilibrium = small_game.find_equilibrium(
        num_producers=3,
        max_iter=200,
        lr=0.1,
        seed=42
    )

    print(f"找到的策略:")
    for i, strategy in enumerate(small_equilibrium['strategies']):
        print(f"  生产者 {i + 1}: {strategy}")

    print(f"最终效用: {small_equilibrium['final_utilities']}")