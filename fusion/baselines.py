# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.baselines
# 用途:     基准排名方法，用于与融合算法对比：
#           - IMDb 贝叶斯加权平均
#           - Massey（封装 utils.rank.MasseyRank）
#           - Keener（封装 utils.rank.KeenerRank）
#           - Offense-Defense（封装 utils.rank.OffenseDefenseRank）
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rank import MasseyRank, KeenerRank, OffenseDefenseRank
from utils.log import Log


def imdb_bayesian_average(csv_path, m=50):
    """
    IMDb 风格贝叶斯加权平均。

    加权分 = (v / (v + m)) * R + (m / (v + m)) * C

    其中：
        v = 该条目的评分人数
        R = 该条目的原始算术平均分
        C = 全站所有条目均分
        m = 进入排名的最低评分人数门槛（收缩强度）

    参数
    ----
    csv_path : str
        条目-用户矩阵 CSV 路径（行=用户，列=条目）。
    m : int
        收缩阈值，默认 50，与 Bangumi 一致。

    返回
    ----
    pd.Series
        按降序排列的条目评分。
    """
    matrix = pd.read_csv(csv_path, index_col=0)
    v = matrix.notna().sum(axis=0)        # 每个条目的评分人数
    R = matrix.mean(axis=0)               # 每个条目的算术均分
    C = R.mean()                          # 全站均分

    weighted = (v / (v + m)) * R + (m / (v + m)) * C
    return weighted.sort_values(ascending=False)


def run_massey(csv_path, update=True):
    """
    在给定 CSV 上运行 Massey 排名。
    返回降序排列的 pd.Series（索引为条目 ID，越高越好）。
    """
    log = Log("massey_baseline").get_log()
    ms = MasseyRank(csv_path, log, update=update)
    ms.solve()
    result = ms.solution_diff.sort_values(ascending=False)
    result.index = result.index.astype(str)
    return result


def run_keener(csv_path, update=True):
    """
    在给定 CSV 上运行 Keener 排名。
    返回降序排列的 pd.Series（索引为条目 ID，越高越好）。
    """
    log = Log("keener_baseline").get_log()
    kn = KeenerRank(csv_path, log, update=update)
    kn.solve()
    result = kn.solution.sort_values(ascending=False)
    result.index = result.index.astype(str)
    return result


def run_od(csv_path):
    """
    在给定 CSV 上运行 Offense-Defense 排名。
    返回降序排列的 pd.Series（索引为条目 ID，越高越好）。
    """
    od = OffenseDefenseRank(csv_path)
    od.solve()
    result = od.solution.sort_values(ascending=False)
    result.index = result.index.astype(str)
    return result
