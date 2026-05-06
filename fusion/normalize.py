# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.normalize
# 用途:     第一层——用户评分归一化。
#           消除用户评分倾向偏差（手松/手紧），并降低无区分度用户的权重。
#
#           z_ui = (s_ui - μ_u) / σ_u
#           w_u  = sigmoid(k * (σ_u - σ_min))
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np


def sigmoid(x):
    """数值稳定的 sigmoid 函数。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def normalize_users(matrix, k=2.0, sigma_min=0.5):
    """
    对用户评分做 z-score 归一化，并计算每位用户的可靠性权重。

    参数
    ----
    matrix : pd.DataFrame
        条目-用户矩阵（行=用户，列=条目），NaN 表示未评分。
    k : float
        方差门控的 sigmoid 陡峭度。k 越大，权重从 0 到 1 的过渡越陡。
    sigma_min : float
        最低可接受的评分标准差。低于此值的用户权重趋近于 0。

    返回
    ----
    Z : pd.DataFrame
        z-score 归一化后的评分矩阵（形状不变，NaN 保留）。
    w_u : pd.Series
        每位用户的权重（索引为用户 ID）。
    user_stats : pd.DataFrame
        列: mu（均值）, sigma（标准差）, weight（权重），供检查。
    """
    # 逐用户计算均值和标准差
    mu = matrix.mean(axis=1)
    sigma = matrix.std(axis=1, ddof=1)   # 样本标准差

    # 标准差为 0（只有 1 次评分或所有评分相同）→ 置 NaN
    sigma = sigma.replace(0, np.nan)

    # z-score 归一化: (x - μ) / σ
    Z = matrix.subtract(mu, axis=0).divide(sigma, axis=0)

    # 用户权重: 评分区分度越高，权重越大
    # sigma 为 NaN 的填 0 → sigmoid(-∞) ≈ 0
    sigma_filled = sigma.fillna(0)
    w_u = sigmoid(k * (sigma_filled - sigma_min))
    w_u = pd.Series(w_u, index=matrix.index, name="weight")

    user_stats = pd.DataFrame({"mu": mu, "sigma": sigma, "weight": w_u})

    return Z, w_u, user_stats
