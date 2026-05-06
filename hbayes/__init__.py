# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     hbayes
# 用途:     层次贝叶斯排名模型。
#           一个联合概率模型替代五层启发式流水线。
#
#           模型:
#             s_ui ~ Normal(q_i + b_u, σ)
#             q_i  ~ Normal(μ_q, τ²_q)   条目质量
#             b_u  ~ Normal(0, τ²_b)     用户偏差
#             μ_q  ~ Normal(7, 2)        全局先验
#             τ_q  ~ HalfNormal(2)
#             τ_b  ~ HalfNormal(2)
#             σ    ~ HalfNormal(2)       观测噪声
#
#           推断: NumPyro SVI (变分推断) + GPU/CPU JAX 后端
#
# 作者:     Atomic
#
# 创建:     2026/5/6
# ------------------------------------------------------------------------------

from hbayes.model import HierarchicalBayesRank, prepare_data

__all__ = ["HierarchicalBayesRank", "prepare_data"]
