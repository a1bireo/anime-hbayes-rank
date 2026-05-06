# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.markov_conduct
# 用途:     第四层——马尔可夫链传递传导。
#           没有共同评分者的条目对也能通过比较网络中的传递路径间接比较。
#
#           从成对比较数据构建 PageRank 风格的转移矩阵，求平稳分布:
#             T_ij   ∝ c_ij * p_ij   （i 给 j 的"投票"权重）
#             T'     = d * T + (1-d) * (1/N)   （阻尼 + 随机跳转）
#             π      = π · T'                   （平稳分布 = 评分）
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd


def markov_conduction(P, C, d=0.85, logger=None):
    """
    通过马尔可夫随机游走在比较网络中传递排名信息。

    原理: 如果 A 胜过 B，B 胜过 C，即使没有人同时评价过 A 和 C，
    A 也应该获得一些来自 C 的间接优势——这就是传递传导。

    参数
    ----
    P : pd.DataFrame
        胜率矩阵（N×N）。P.loc[i, j] = i 胜过 j 的后验概率。
    C : pd.DataFrame
        置信度矩阵（N×N）。
    d : float
        阻尼因子（0 < d < 1）。d=0.85 是 PageRank 标准值。
        d 越低 → 越接近均匀分布（传导效果弱）；
        d 越高 → 对网络结构越敏感（传导效果强）。
    logger : logging.Logger 或 None

    返回
    ----
    pd.Series
        经传递传导调整后的评分（索引为条目 ID），越高越好。
    """
    N = len(P)
    subjects = P.index

    P_vals = P.values
    C_vals = C.values

    # T_raw[i, j]: 从 i 到 j 的边权重（i 把票投给 j）
    # 当 j 胜过 i 时发生投票，权重 = P(j 胜过 i) * C(j, i)
    T_raw = P_vals.T * C_vals.T

    # 去掉自环
    np.fill_diagonal(T_raw, 0)

    # 行归一化 → 随机矩阵
    row_sums = T_raw.sum(axis=1, keepdims=True)
    # 没有任何出边的行 → 均匀分布
    T = np.where(row_sums > 0, T_raw / row_sums, 1.0 / N)

    # 施加阻尼（随机跳转）
    T_damped = d * T + (1 - d) / N * np.ones((N, N))

    # 幂迭代求平稳分布
    pi = np.ones(N) / N
    for iteration in range(200):
        pi_next = pi @ T_damped
        delta = np.abs(pi_next - pi).max()
        pi = pi_next
        if delta < 1e-12:
            if logger:
                logger.debug("马尔可夫传导在第 {} 轮收敛".format(iteration + 1))
            break

    return pd.Series(pi, index=subjects, name="markov_rating")
