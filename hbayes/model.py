# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     hbayes.model
# 用途:     层次贝叶斯排名模型。
#
#           观测:  s_ui ~ Normal(q_i + b_u, σ_u)
#           条目:  q_i  ~ Normal(μ_q, τ_q / √(1 + v_i / n_ref))
#                  精度加权先验——评分人数少则先验收紧，自动向全局均值收缩
#                  n_ref = 评分人数中位数（纯数据驱动，零人工参数）
#           用户:  b_u  ~ Normal(0, τ_b)
#                  σ_u  ~ HalfNormal(1)  每用户噪声
#           超参:  μ_q  ~ Normal(7, 2)
#                  τ_q  ~ HalfNormal(2)
#                  τ_b  ~ HalfNormal(2)
#
# 作者:     Atomic
#
# 创建:     2026/5/6
# 更新:     2026/5/6 — 回退 Normal + 精度加权先验
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam


def prepare_data(csv_path):
    """
    将条目-用户矩阵转换为 NumPyro 长格式。

    返回
    ----
    dict 包含 obs, user_idx, subject_idx, vote_counts, 维度信息, 原始ID列表
    """
    matrix = pd.read_csv(csv_path, index_col=0)

    user_ids = list(matrix.index)
    subject_ids = list(matrix.columns)
    user_map = {u: i for i, u in enumerate(user_ids)}
    subject_map = {s: i for i, s in enumerate(subject_ids)}

    obs_list = []
    user_idx_list = []
    subject_idx_list = []

    for u in user_ids:
        u_idx = user_map[u]
        row = matrix.loc[u].dropna()
        for s, rating in row.items():
            obs_list.append(float(rating))
            user_idx_list.append(u_idx)
            subject_idx_list.append(subject_map[s])

    # 每个条目的评分人数
    vc = matrix.notna().sum(axis=0)
    vc_indexed = [int(vc[s]) for s in subject_ids]

    return {
        "obs": jnp.array(obs_list, dtype=jnp.float32),
        "user_idx": jnp.array(user_idx_list, dtype=jnp.int32),
        "subject_idx": jnp.array(subject_idx_list, dtype=jnp.int32),
        "vote_counts": jnp.array(vc_indexed, dtype=jnp.float32),
        "N_users": len(user_ids),
        "N_subjects": len(subject_ids),
        "user_ids": user_ids,
        "subject_ids": subject_ids,
    }


def model(data):
    """
    层次贝叶斯排名模型。

    s_ui ~ Normal(q_i + b_u, σ_u)
    q_i  ~ Normal(μ_q, τ_q / √(1 + v_i / n_ref))
        n_ref = median(v_i)，评分人数越少，先验越紧
    b_u  ~ Normal(0, τ_b)
    σ_u  ~ HalfNormal(1)
    """
    N_users = data["N_users"]
    N_subjects = data["N_subjects"]
    v = data["vote_counts"]

    # n_ref = 评分人数中位数，纯数据驱动
    n_ref = jnp.median(v)

    # 超先验
    mu_q = numpyro.sample("mu_q", dist.Normal(7.0, 2.0))
    tau_q = numpyro.sample("tau_q", dist.HalfNormal(2.0))
    tau_b = numpyro.sample("tau_b", dist.HalfNormal(2.0))

    # 精度加权先验: 评分人数少 → 先验标准差小 → 向 μ_q 收缩更强
    # v_i → 0:  prior_scale → tau_q（基准收缩）
    # v_i = n_ref: prior_scale → tau_q / √2 ≈ 0.7 τ_q（中等收缩）
    # v_i → ∞:  prior_scale → 0（完全放开，靠数据说话）
    # 注意: 这里 v_i 越大 → scale 越小 → 先验越紧 → 收缩越强
    #       这是反直觉的。正确做法应该是:
    #       prior_scale = tau_q * √(1 + n_ref / v_i) 不对...
    #
    # 正确的精度加权: 先验精度 = 1/τ_q² + v_i/σ²
    # 当我们没有独立的数据精度的估计时，用:
    #   先验标准差 ∝ 1/√(基线 + 数据量)
    #   即 prior_scale = τ_q / √(1 + v_i / n_ref)
    #
    #   v_i = 0:    prior_scale = τ_q       （只有先验）
    #   v_i → ∞:    prior_scale → 0         （数据无限精确）
    #
    # 等等，又弄反了。我们想要的是 v_i 大 → 先验弱（相信数据），v_i 小 → 先验强（拉回均值）。
    # prior_scale = τ_q * √(1 + n_ref / max(v_i, 1))
    #   v_i = 1:    prior_scale ≈ τ_q * √(n_ref)  先验很宽 → 弱收缩 → 不对！
    #
    # 直接用 precision 框架:
    # 后验精度 = 先验精度 + 数据精度
    # 先验精度 = 1/τ_q²（对所有条目相同）
    # 数据精度 ≈ v_i / σ²_avg（评分越多越精确）
    # 后验均值 ≈ (先验精度 * μ_q + 数据精度 * sample_mean) / (先验精度 + 数据精度)
    # 收缩权重 = 先验精度 / (先验精度 + 数据精度)
    #
    # 当 v_i 小: 数据精度低 → 收缩权重大 → 后验靠近 μ_q ✓
    # 当 v_i 大: 数据精度高 → 收缩权重小 → 后验靠近 sample_mean ✓
    #
    # 先驗嵌入: q_i ~ Normal(μ_q, τ_q)
    #   然后似然中有 n_i 条独立观测 → 自动产生上述收缩
    #   不需要修改 q_i 的先验!  层次模型已经天然做到了!
    #
    # 但当前的问题是: SVI 对低 n_i 条目学到的后验方差仍然很窄
    # 因为 AutoNormal 的 scale 参数可能欠估计
    #
    # 所以我们手动加强低 n_i 条目的先验:
    #   q_i ~ Normal(μ_q, τ_q * sqrt(n_ref / (n_ref + v_i)))
    #   这才是对的: v_i 大 → sqrt(小) → 先验紧 → 收缩弱(相信数据)
    #   不对... sqrt(n_ref/(n_ref+v_i)) → v_i大则值小 → 先验标准差小 → 收缩强
    #   还是反了!
    #
    # 换个思路: 用数据精度做权重
    #   weight_i = v_i / (v_i + n_ref)   ∈ [0, 1]
    #   q_i ~ Normal(μ_q, τ_q * √(1 - weight_i * relaxation))
    #   不做这个。最简单的正确做法:
    #
    # q_i 的先验标准差应当随 v_i 增大而增大 (v_i大 → 相信数据 → 先验松)
    # 即: prior_scale_i = τ_q * √(v_i / (v_i + n_ref))
    #   v_i=0:   scale=0          → 完全收缩到 μ_q
    #   v_i=n_ref: scale=τ_q/√2   → 中等
    #   v_i→∞:   scale→τ_q        → 放开
    # 精度加权: v_i < n_ref 的先验收紧，v_i >= n_ref 的不额外限制
    # weight ∈ [min_v/n_ref, 1]，线性缩放
    weight = jnp.clip(v / n_ref, 0.0, 1.0)
    prior_scale = tau_q * weight

    # 条目质量
    with numpyro.plate("subjects", N_subjects):
        q = numpyro.sample("q", dist.Normal(mu_q, prior_scale))

    # 用户偏差 + 每用户噪声
    with numpyro.plate("users", N_users):
        b = numpyro.sample("b", dist.Normal(0.0, tau_b))
        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(1.0))

    # 观测
    mu_obs = q[data["subject_idx"]] + b[data["user_idx"]]
    sigma_per_obs = sigma_u[data["user_idx"]]

    with numpyro.plate("observations", len(data["obs"])):
        numpyro.sample("s", dist.Normal(mu_obs, sigma_per_obs), obs=data["obs"])


class HierarchicalBayesRank:
    """
    层次贝叶斯排名器。

    参数
    ----
    csv_path : str  条目-用户矩阵 CSV 路径
    rng_seed : int  随机种子
    """

    def __init__(self, csv_path, rng_seed=42):
        self.csv_path = csv_path
        self.rng = jrandom.PRNGKey(rng_seed)
        self.data = prepare_data(csv_path)
        self.guide = None
        self.svi_result = None
        self.ranking_ = None

    def fit(self, steps=15000, lr=0.003, verbose=True):
        """SVI 拟合。"""
        optimizer = Adam(step_size=lr)
        self.guide = AutoNormal(model)
        svi = SVI(model, self.guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(self.rng, steps, self.data, progress_bar=verbose)
        self.svi_result = svi_result

        posterior_median = self.guide.median(svi_result.params)
        q_median = np.array(posterior_median["q"])
        b_median = np.array(posterior_median["b"])
        sigma_u_median = np.array(posterior_median["sigma_u"])

        self.ranking_ = pd.Series(
            q_median, index=self.data["subject_ids"], name="hb_score"
        ).sort_values(ascending=False)

        self.user_bias_ = pd.Series(
            b_median, index=self.data["user_ids"], name="user_bias"
        )
        self.user_noise_ = pd.Series(
            sigma_u_median, index=self.data["user_ids"], name="user_noise"
        )

        return self

    def rank(self, top_n=None):
        if self.ranking_ is None:
            raise RuntimeError("请先调用 fit()")
        if top_n is not None:
            return self.ranking_.head(top_n)
        return self.ranking_

    def save(self, output_dir="solution"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.ranking_.to_csv(os.path.join(output_dir, "hb_ranking.csv"))
        if hasattr(self, "user_bias_"):
            self.user_bias_.to_csv(os.path.join(output_dir, "hb_user_bias.csv"))
        if hasattr(self, "user_noise_"):
            self.user_noise_.to_csv(os.path.join(output_dir, "hb_user_noise.csv"))
