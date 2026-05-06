# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.bradley_terry
# 用途:     第三层——Bradley-Terry 模型求解器。
#           将成对胜率矩阵转化为全局评分向量，使用加权正则化最小二乘。
#
#           Bradley-Terry 模型:  P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j))
#           →  log(p_ij / (1 - p_ij)) = r_i - r_j
#
#           求解超定线性系统:
#             D · r = t
#           每行 D 在 i 列 = +1, j 列 = -1, t_k = logit(p_ij)。
#
#           以置信度 c_ij 加权，L2 正则化 λ:
#             r = (DᵀWD + λI)⁻¹ DᵀWt
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd


def logit(p, eps=0.02):
    """
    log(p / (1-p))，截断以防极端概率主导模型。

    截断到 [eps, 1-eps]：
    - logit(0.98) = 3.9, logit(0.99) = 4.6, logit(0.999) = 6.9
    - 不做截断时，p→1 的边贡献趋近无穷大，会完全主导最小二乘拟合
    - eps=0.02 意味着单条边最多贡献 logit(0.98)≈3.9，相当于约 8 条 p=0.7 的边
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def solve_bradley_terry(P, C, lam=1.0, logger=None):
    """
    通过加权正则化最小二乘求解 Bradley-Terry 模型。

    参数
    ----
    P : pd.DataFrame
        胜率矩阵（N×N）。P.loc[i, j] = i 胜过 j 的后验概率。
    C : pd.DataFrame
        置信度矩阵（N×N）。值越高表示该对比较越可靠。
    lam : float
        L2 正则化强度。防止共评人数少的条目对过拟合。
        lam 越大 → 评分越接近 0。
    logger : logging.Logger 或 None

    返回
    ----
    pd.Series
        每个条目的评分（索引为条目 ID），零中心化。
    """
    subjects = P.index
    N = len(subjects)

    # 取上三角 (i < j)，避免重复计数
    i_idx, j_idx = np.triu_indices(N, k=1)

    p_vals = P.values[i_idx, j_idx]  # P(i 胜过 j)
    c_vals = C.values[i_idx, j_idx]  # 该对的置信度

    # 仅保留有共同评分者的条目对
    valid = c_vals > 0
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    p_vals = p_vals[valid]
    c_vals = c_vals[valid]
    E = len(i_idx)

    if logger:
        logger.info("Bradley-Terry: {} 个有效条目对（总计 {} 个可能组合）"
                    .format(E, N * (N - 1) // 2))

    # 目标向量: 胜率做 logit 变换
    t = logit(p_vals)

    # 直接组装 DᵀWD 和 DᵀWt，无需显式构建巨型稀疏矩阵 D(E×N)。
    #
    # D 的形状是 (E, N)，第 k 行对应条目对 (i, j):
    #   D[k, i] = +1, D[k, j] = -1, 其余为 0
    #
    # DᵀWD 为 N×N，其中:
    #   (DᵀWD)[i, i] += w_k       对每对涉及 i 的比较
    #   (DᵀWD)[j, j] += w_k       对每对涉及 j 的比较
    #   (DᵀWD)[i, j] -= w_k       交叉项
    #   (DᵀWD)[j, i] -= w_k       对称
    #
    # DᵀWt 为 (N,)，其中:
    #   (DᵀWt)[i] += w_k * t_k    i 为 +1 的边
    #   (DᵀWt)[j] -= w_k * t_k    j 为 -1 的边

    DtWD = np.zeros((N, N), dtype=np.float64)
    DtWt = np.zeros(N, dtype=np.float64)

    for k in range(E):
        ii, jj = i_idx[k], j_idx[k]
        w = c_vals[k]
        tk = t[k]

        # 对角贡献
        DtWD[ii, ii] += w
        DtWD[jj, jj] += w
        # 非对角
        DtWD[ii, jj] -= w
        DtWD[jj, ii] -= w
        # 右边项
        DtWt[ii] += w * tk
        DtWt[jj] -= w * tk

    # L2 正则化
    DtWD += lam * np.eye(N)

    # 添加零中心约束: Σ r_i = 0
    # 在矩阵底部追加一行全 1，右边追加 0
    A_aug = np.vstack([DtWD, np.ones(N)])
    b_aug = np.append(DtWt, 0.0)

    # 最小二乘求解（增广后系统略为超定）
    r, residuals, rank, sv = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    if logger:
        logger.info("Bradley-Terry 求解完成: 秩={}, 残差={:.6f}"
                    .format(rank, residuals[0] if len(residuals) > 0 else 0))

    return pd.Series(r, index=subjects, name="bt_rating")
