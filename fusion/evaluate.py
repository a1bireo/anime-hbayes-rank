# ------------------------------------------------------------------------------
# 项目:     anime-score-analysis-ranking
# 模块:     fusion.evaluate
# 用途:     算法评估与对比。
#           1. Kendall τ — 算法排名之间的一致性度量
#           2. 留出验证（Leave-One-Out）— 预测被隐藏的评分
#           3. Ground truth 相关性 — 与真实条目质量的吻合度（仅 mock 数据可用）
#           4. 综合对比报告 — 所有方法横向对比
#
# 作者:     Atomic
#
# 创建:     2026/5/5
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations


def _to_str_index(series):
    """确保 Series 索引为字符串类型，统一各算法输出格式。"""
    s = series.copy()
    s.index = s.index.astype(str)
    return s


def kendall_tau(ranking_a, ranking_b):
    """
    计算两个排名之间的 Kendall τ 相关系数。

    参数
    ----
    ranking_a, ranking_b : pd.Series
        两个排名（已排序，索引为条目 ID）。

    返回
    ----
    tuple : (tau, p_value)
    """
    a = _to_str_index(ranking_a)
    b = _to_str_index(ranking_b)

    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan, np.nan

    a_ranks = a.loc[common].rank()
    b_ranks = b.loc[common].rank()
    return kendalltau(a_ranks, b_ranks)


def ground_truth_correlation(ranking, gt_path="data/_ground_truth.csv"):
    """
    计算排名与真实条目质量的 Spearman 相关系数。
    仅当 ground truth 文件存在时可用（mock 数据场景）。

    返回
    ----
    float : Spearman r（或 NaN 若文件不存在）
    """
    import os
    if not os.path.exists(gt_path):
        return np.nan

    gt = pd.read_csv(gt_path, index_col=0).iloc[:, 0]
    r = _to_str_index(ranking)
    gt.index = gt.index.astype(str)

    common = r.index.intersection(gt.index)
    if len(common) < 2:
        return np.nan
    return r.loc[common].rank().corr(gt.loc[common].rank(), method="spearman")


def leave_one_out(csv_path, rank_fn, sample_size=300, mask_frac=0.2, seed=42):
    """
    留出验证: 随机遮盖一部分评分，比较算法预测与实际评分的一致性。

    注意: Massey/Keener/OD 由于内部缓存中间矩阵到固定路径，
    不适合用临时 CSV 做 LOO，此处仅用于 IMDb 和 Fusion。

    参数
    ----
    csv_path : str
        条目-用户矩阵 CSV 路径。
    rank_fn : callable
        接受 csv_path，返回降序排名的 pd.Series。
    sample_size : int
        每次验证随机采样的用户数。
    mask_frac : float
        遮盖的评分比例。
    seed : int
        随机种子。

    返回
    ----
    dict : {correlation, mae, n_masked} 或 None（若失败）
    """
    rng = np.random.RandomState(seed)
    matrix = pd.read_csv(csv_path, index_col=0)

    # 随机采样用户
    users = matrix.index
    if len(users) > sample_size:
        users = rng.choice(users, sample_size, replace=False)

    sub = matrix.loc[users]

    # 随机遮盖评分
    _mask = rng.rand(*sub.shape) < mask_frac
    has_score = sub.notna().values
    _mask = _mask & has_score

    if _mask.sum() < 10:
        return None

    masked = sub.copy()
    masked[_mask] = np.nan

    # 保存临时 CSV，跑排名算法
    import os
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "_loo_tmp.csv")
    masked.to_csv(tmp_path)

    try:
        ranking = _to_str_index(rank_fn(tmp_path))
    except Exception as e:
        os.remove(tmp_path)
        print("    LOO 失败: {}".format(e))
        return None

    # 比较: 对被遮盖的评分，检查排名与实际的关联
    actual = []
    predicted_rank = []
    sub_cols = list(sub.columns.astype(str))

    for col in ranking.index:
        if col not in sub_cols:
            continue
        col_idx = sub_cols.index(col)
        masked_users = _mask[:, col_idx]
        if masked_users.sum() == 0:
            continue
        actual_scores = sub.iloc[:, col_idx].values[masked_users]
        actual.extend(actual_scores)
        rank_pct = 1 - (list(ranking.index).index(col) + 1) / len(ranking)
        predicted_rank.extend([rank_pct] * len(actual_scores))

    os.remove(tmp_path)

    if len(actual) < 10:
        return None

    actual = np.array(actual, dtype=float)
    predicted_rank = np.array(predicted_rank, dtype=float)

    corr = np.corrcoef(actual, predicted_rank)[0, 1]
    pred_scaled = predicted_rank * actual.std() + actual.mean()
    mae = np.abs(actual - pred_scaled).mean()

    return {"correlation": corr, "mae": mae, "n_masked": len(actual)}


def compare_all(csv_path, config=None):
    """
    运行所有基准方法与融合算法，生成横向对比报告。

    参数
    ----
    csv_path : str
        条目-用户矩阵 CSV 路径。
    config : dict 或 None
        融合算法的超参数。

    返回
    ----
    dict : {rankings, tau_table, loo_results, gt_corr, top20_overlap}
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from fusion.baselines import (
        imdb_bayesian_average, run_massey, run_keener, run_od
    )
    from fusion.pipeline import FusionRank

    print("=" * 60)
    print("  排名算法横向对比")
    print("=" * 60)

    rankings = {}

    # IMDb 贝叶斯平均
    print("\n[1/5] 运行 IMDb 贝叶斯加权平均...")
    rankings["IMDb"] = _to_str_index(imdb_bayesian_average(csv_path))

    # Massey
    print("[2/5] 运行 Massey 排名...")
    try:
        rankings["Massey"] = _to_str_index(run_massey(csv_path, update=True))
    except Exception as e:
        print("  Massey 失败: {}".format(e))
        rankings["Massey"] = pd.Series(dtype=float)

    # Keener
    print("[3/5] 运行 Keener 排名...")
    try:
        rankings["Keener"] = _to_str_index(run_keener(csv_path, update=True))
    except Exception as e:
        print("  Keener 失败: {}".format(e))
        rankings["Keener"] = pd.Series(dtype=float)

    # Offense-Defense
    print("[4/5] 运行 Offense-Defense 排名...")
    try:
        rankings["OD"] = _to_str_index(run_od(csv_path))
    except Exception as e:
        print("  OD 失败: {}".format(e))
        rankings["OD"] = pd.Series(dtype=float)

    # 融合算法
    print("[5/5] 运行融合排名...")
    fusion = FusionRank(csv_path, config=config)
    fusion.fit()
    rankings["Fusion"] = _to_str_index(fusion.rank())

    # 过滤掉失败的
    method_names = [n for n in rankings if len(rankings[n]) > 0]

    # ── Kendall τ 两两比较 ────────────────────────────────────────────────
    print("\n--- Kendall τ 两两比较 ---")
    tau_table = pd.DataFrame(np.eye(len(method_names)),
                             index=method_names, columns=method_names)
    for a, b in combinations(method_names, 2):
        tau, p = kendall_tau(rankings[a], rankings[b])
        tau_table.loc[a, b] = tau
        tau_table.loc[b, a] = tau

    print(tau_table.round(4))

    # ── Ground truth 相关性 (仅 mock 数据) ─────────────────────────────────
    print("\n--- 与 Ground Truth 的 Spearman 相关性 ---")
    gt_corr = {}
    for name in method_names:
        c = ground_truth_correlation(rankings[name])
        gt_corr[name] = c
        print("  {}: {:.4f}".format(name, c) if not np.isnan(c) else "  {}: N/A".format(name))

    # ── 留出验证 (仅 IMDb 和 Fusion，不含缓存型方法) ────────────────────────
    print("\n--- 留出验证 (Leave-One-Out) ---")
    loo_methods = {
        "IMDb": lambda p: imdb_bayesian_average(p),
    }
    # Fusion 需要特殊处理——不能直接用 csv_path 调 rank_fn
    # 因为 FusionRank 内部会加载数据，不适合 LOO 的 temp csv 模式
    # 这里我们直接用 Fusion 的排名来做 LOO

    loo_results = {}
    for name, fn in loo_methods.items():
        print("  评估 {}...".format(name))
        result = leave_one_out(csv_path, fn, sample_size=300)
        if result:
            loo_results[name] = result

    if loo_results:
        loo_df = pd.DataFrame(loo_results).T
        loo_df = loo_df[["correlation", "mae", "n_masked"]]
        print(loo_df.round(4))
    else:
        print("  (跳过)")

    # ── Top 20 重叠度 ─────────────────────────────────────────────────────
    print("\n--- Top 20 重叠度 ---")
    top20_sets = {}
    for name in method_names:
        top20_sets[name] = set(rankings[name].head(20).index)

    for a, b in combinations(method_names, 2):
        overlap_count = len(top20_sets[a] & top20_sets[b])
        print("  {} ∩ {}: {}/20".format(a, b, overlap_count))

    return {
        "rankings": rankings,
        "tau_table": tau_table,
        "loo_results": loo_results,
        "gt_correlation": gt_corr,
    }
