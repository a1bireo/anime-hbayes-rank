# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.pipeline
# 用途:     编排五层融合排名流水线。
#
#   第一层: normalize_users()        — 用户 z-score 归一化 + 方差门控
#   第二层: build_pairwise_matrix()  — 成对胜率 + 贝叶斯收缩
#   第三层: solve_bradley_terry()    — 加权正则化最小二乘求解
#   第四层: markov_conduction()      — PageRank 传递传导
#        → 混合: α * BT + (1-α) * MC
#   第五层: bayesian_shrinkage()     — IMDb 风格全局收缩
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.normalize import normalize_users
from fusion.pairwise import build_pairwise_matrix
from fusion.bradley_terry import solve_bradley_terry
from fusion.markov_conduct import markov_conduction
from fusion.shrinkage import bayesian_shrinkage
from utils.log import Log


class FusionRank:
    """
    多层融合排名算法。

    融合了以下方法的核心思想：
    - Massey   → 最小二乘框架（第三层）
    - Keener   → 成对比较 + 网络传导（第二层 → 第三层/四层）
    - Colley   → 先验内嵌于矩阵结构（第二层贝叶斯收缩）
    - Elo      → 逻辑斯蒂期望（第三层 Bradley-Terry）
    - GeM      → 随机游走传递传导（第四层）
    - IMDb     → 贝叶斯全局收缩（第五层）
    - O-D      → 用户差异化（第一层归一化 + 权重门控）

    参数
    ----
    csv_path : str
        条目-用户矩阵 CSV 路径（行=用户，列=条目，值=评分，空=未评分）。
    config : dict 或 None
        超参数覆盖。可用键见 DEFAULT_CONFIG。

    属性
    ----
    ranking_ : pd.Series
        fit() 后的最终排名，降序排列。
    layers_ : dict
        各层中间结果，供调试和检查。
    """

    DEFAULT_CONFIG = {
        # 第一层: 归一化
        "k": 2.0,           # 方差门控的 sigmoid 陡峭度
        "sigma_min": 0.5,   # 最低可接受评分标准差

        # 第二层: 成对比较
        "m_0": 5,           # 贝叶斯先验强度（凭空想象的平局场数）
        "soft_win": False,  # 硬胜负（True = sigmoid 软胜负）

        # 第三层: Bradley-Terry
        "lam": 1.0,         # L2 正则化强度

        # 第四层: 马尔可夫传导
        "d": 0.85,          # 阻尼因子（PageRank 标准值）
        "alpha": 0.7,       # 混合权重: α * BT + (1-α) * MC

        # 第五层: 收缩
        "M": 50,            # 全局收缩阈值
    }

    def __init__(self, csv_path, config=None):
        self.csv_path = csv_path
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self.logger = Log("fusion").get_log()
        self.matrix_raw = None    # 原始条目-用户矩阵
        self.layers_ = {}         # 各层中间结果
        self.ranking_ = None      # 最终排名
        self.vote_counts_ = None  # 每个条目的评分人数

    def fit(self):
        """依次执行全部五层流水线。"""
        t0 = time.time()

        # ── 加载数据 ──────────────────────────────────────────────────────
        self.logger.info("加载数据: {}".format(self.csv_path))
        self.matrix_raw = pd.read_csv(self.csv_path, index_col=0)
        self.vote_counts_ = self.matrix_raw.notna().sum(axis=0)
        self.logger.info("已加载: {} 用户 × {} 条目"
                         .format(*self.matrix_raw.shape))

        # ── 第一层: 用户归一化 ─────────────────────────────────────────────
        self.logger.info("第一层: 用户评分归一化")
        Z, w_u, user_stats = normalize_users(
            self.matrix_raw,
            k=self.config["k"],
            sigma_min=self.config["sigma_min"],
        )
        self.layers_["Z"] = Z
        self.layers_["user_weights"] = w_u
        self.layers_["user_stats"] = user_stats
        self.logger.info("  用户权重: 均值={:.3f}, 最小={:.3f}, 最大={:.3f}"
                         .format(w_u.mean(), w_u.min(), w_u.max()))

        # ── 第二层: 成对比较矩阵 ───────────────────────────────────────────
        self.logger.info("第二层: 构建成对比较矩阵")
        P, C = build_pairwise_matrix(
            Z,
            m_0=self.config["m_0"],
            soft_win=self.config["soft_win"],
            logger=self.logger,
        )
        self.layers_["P"] = P
        self.layers_["C"] = C
        self.logger.info("  成对矩阵: {}×{}, 稀疏度={:.1%}"
                         .format(P.shape[0], P.shape[1],
                                 1 - (C.values > 0).mean()))

        # ── 第三层: Bradley-Terry 求解 ─────────────────────────────────────
        self.logger.info("第三层: Bradley-Terry 模型求解")
        r_bt = solve_bradley_terry(
            P, C,
            lam=self.config["lam"],
            logger=self.logger,
        )
        self.layers_["r_bt"] = r_bt
        self.logger.info("  BT 评分: 均值={:.4f}, 标准差={:.4f}, 范围=[{:.4f}, {:.4f}]"
                         .format(r_bt.mean(), r_bt.std(), r_bt.min(), r_bt.max()))

        # ── 第四层: 马尔可夫传递传导 ────────────────────────────────────────
        self.logger.info("第四层: 马尔可夫传递传导 (d={})"
                         .format(self.config["d"]))
        r_mc = markov_conduction(
            P, C,
            d=self.config["d"],
            logger=self.logger,
        )
        self.layers_["r_mc"] = r_mc

        # 混合 Bradley-Terry 与马尔可夫传导
        alpha = self.config["alpha"]
        r_blend = alpha * r_bt + (1 - alpha) * r_mc
        self.layers_["r_blend"] = r_blend
        self.logger.info("  混合 (α={:.2f}): BT×{:.2f} + MC×{:.2f}"
                         .format(alpha, alpha, 1 - alpha))

        # ── 第五层: 贝叶斯收缩 ─────────────────────────────────────────────
        self.logger.info("第五层: 全局贝叶斯收缩 (M={})"
                         .format(self.config["M"]))
        self.ranking_ = bayesian_shrinkage(
            r_blend,
            self.vote_counts_,
            M=self.config["M"],
        )
        self.logger.info("  最终排名: {} 个条目".format(len(self.ranking_)))

        elapsed = time.time() - t0
        self.logger.info("流水线执行完毕，耗时 {:.1f} 秒".format(elapsed))

        return self

    def rank(self, top_n=None):
        """返回排名结果（降序排列的 pd.Series）。"""
        if self.ranking_ is None:
            raise RuntimeError("请先调用 fit()。")
        if top_n is not None:
            return self.ranking_.head(top_n)
        return self.ranking_

    def save(self, output_dir="solution"):
        """将所有层输出保存为 CSV 文件。"""
        os.makedirs(output_dir, exist_ok=True)

        self.ranking_.to_csv(os.path.join(output_dir, "fusion_ranking.csv"))

        if "r_bt" in self.layers_:
            self.layers_["r_bt"].sort_values(ascending=False) \
                .to_csv(os.path.join(output_dir, "fusion_bt.csv"))
        if "r_mc" in self.layers_:
            self.layers_["r_mc"].sort_values(ascending=False) \
                .to_csv(os.path.join(output_dir, "fusion_markov.csv"))
        if "user_stats" in self.layers_:
            self.layers_["user_stats"].to_csv(
                os.path.join(output_dir, "fusion_user_stats.csv"))

        self.logger.info("结果已保存到 {}/".format(output_dir))
