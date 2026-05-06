# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.pairwise
# 用途:     第二层——从归一化评分构建成对比较矩阵。
#           对每对条目 (i, j) 计算：
#             n_ij  = 同时评价过 i 和 j 的用户数
#             p_ij  = 在这些共同评价者中，给 i 评分高于 j 的比例（经贝叶斯收缩）
#             c_ij  = 置信度 = n_ij / (n_ij + m_0)
#
#           使用 numpy 广播向量化（外层循环遍历条目，内层完全向量化），
#           与现有 Massey/Keener 的 matrix_initial 方法保持一致的模式。
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np


def build_pairwise_matrix(Z, m_0=5, soft_win=False, logger=None):
    """
    构建胜率矩阵和置信度矩阵。

    参数
    ----
    Z : pd.DataFrame
        归一化评分矩阵（行=用户，列=条目），NaN 表示未评分。
    m_0 : int
        贝叶斯先验强度。可以理解为"每对条目凭空想象的平局场数"。
        m_0 越大，需要越多共同评分者才能推翻 0.5 的先验。
    soft_win : bool
        若为 True，用 sigmoid 软胜负代替硬胜负（z_i > z_j）。
        硬胜负更简单，通常足够。
    logger : logging.Logger 或 None

    返回
    ----
    P : pd.DataFrame
        胜率矩阵（N×N，行列均为条目 ID）。
        P.loc[i, j] = 在成对比较中 i 胜过 j 的后验概率。
    C : pd.DataFrame
        置信度矩阵（N×N）。C.loc[i, j] ∈ [0, 1]，越高表示共同评分人数越多。
    """
    # 转置为 (条目数, 用户数) 以便列间广播
    A = Z.T.to_numpy(dtype=np.float64)         # (N_subjects, N_users)
    mask = ~np.isnan(A)                         # (N_subjects, N_users)

    N = A.shape[0]
    P_raw = np.zeros((N, N), dtype=np.float64)
    N_co = np.zeros((N, N), dtype=np.float64)

    # 外层遍历条目 i —— 与已有 utils/rank.py 中的模式一致
    for i in range(N):
        # mask_i: (1, U), mask: (N, U) → co_mask: (N, U)
        mask_i = mask[i, np.newaxis, :]
        co_mask = mask_i & mask                 # 哪些用户同时评价了 i 和 j
        n_ij = co_mask.sum(axis=1)              # (N,) 共同评分人数

        # 条目 i vs 所有条目 j 的胜负
        score_i = A[i, np.newaxis, :]           # (1, U)
        if soft_win:
            # sigmoid 软胜负: P(i > j) = 1 / (1 + exp(-(z_i - z_j) / s))
            diff = (score_i - A) / 0.5
            diff = np.clip(diff, -50, 50)
            soft = 1.0 / (1.0 + np.exp(-diff))
            wins = np.where(co_mask, soft, 0.0).sum(axis=1)
        else:
            wins = ((score_i > A) & co_mask).sum(axis=1)

        P_raw[i, :] = wins
        N_co[i, :] = n_ij

        if logger and (i + 1) % 50 == 0:
            logger.debug("成对比较: 已完成 {}/{} 行".format(i + 1, N))

    # 贝叶斯收缩: p_ij = (n_ij * p_raw + m_0 * 0.5) / (n_ij + m_0)
    # 拉普拉斯先验：每对条目凭空有 m_0 场"平局"
    with np.errstate(divide='ignore', invalid='ignore'):
        P_shrunk = np.where(
            N_co > 0,
            (N_co * (P_raw / np.maximum(N_co, 1)) + m_0 * 0.5) / (N_co + m_0),
            0.5  # 无共同评分者 → 均匀先验
        )

    # 置信度: 随 n_ij 增大逐渐饱和到 1
    C_conf = N_co / (N_co + m_0)

    # 构建 DataFrame
    subjects = Z.columns
    P = pd.DataFrame(P_shrunk, index=subjects, columns=subjects)
    C = pd.DataFrame(C_conf, index=subjects, columns=subjects)

    return P, C
