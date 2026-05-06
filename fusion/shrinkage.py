# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.shrinkage
# 用途:     第五层——全局贝叶斯收缩（IMDb 公式）。
#           将少人评分条目的分数向全局均值拉回，防止小样本极端值。
#
#           r_final[i] = (v_i / (v_i + M)) * r[i] + (M / (v_i + M)) * r_global
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np


def bayesian_shrinkage(ratings, vote_counts, M=50):
    """
    对评分向量施加 IMDb 风格贝叶斯收缩。

    参数
    ----
    ratings : pd.Series
        每个条目的原始评分（索引为条目 ID）。
    vote_counts : pd.Series
        每个条目的评分人数（相同索引）。
    M : int
        收缩阈值。低于 M 人评分的条目会被强力拉向全局均值。

    返回
    ----
    pd.Series
        收缩后的评分，降序排列。
    """
    r_global = ratings.mean()           # 全局均值
    v = vote_counts.astype(float)

    shrunk = (v / (v + M)) * ratings + (M / (v + M)) * r_global
    shrunk.name = "fusion_score"
    return shrunk.sort_values(ascending=False)
