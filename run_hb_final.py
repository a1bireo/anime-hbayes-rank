"""最终运行：精度加权层次贝叶斯 + 生成 web 数据"""
import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
from hbayes.model import HierarchicalBayesRank
from fusion.baselines import imdb_bayesian_average
from fusion.evaluate import kendall_tau, _to_str_index

csv = 'data/_subject_user_matrix.csv'

print('=== 精度加权层次贝叶斯 ===')
hb = HierarchicalBayesRank(csv, rng_seed=42)
d = hb.data
n_ref = float(np.median(np.array(d['vote_counts'])))
print(f'n_ref (评分人数中位数) = {n_ref:.0f}')
print(f'items={d["N_subjects"]} users={d["N_users"]} ratings={len(d["obs"]):,}')

t0 = time.time()
hb.fit(steps=15000, lr=0.003, verbose=False)
elapsed = time.time() - t0
losses = hb.svi_result.losses
print(f'SVI: {elapsed:.0f}s loss={losses[-1]:.0f}')

# 对比 IMDb
r_hb = _to_str_index(hb.rank())
r_imdb = _to_str_index(imdb_bayesian_average(csv))
tau, _ = kendall_tau(r_hb, r_imdb)
overlap20 = len(set(r_hb.head(20).index) & set(r_imdb.head(20).index))
print(f'tau={tau:.4f}  top20={overlap20}/20')

# 归一化到 1-10
q_raw = hb.ranking_.copy()
b_median = hb.user_bias_.median()
predicted = q_raw + b_median
p_min, p_max = predicted.min(), predicted.max()
q_scaled = 1 + 9 * (predicted - p_min) / (p_max - p_min)
q_scaled = q_scaled.sort_values(ascending=False)

# 元数据
subj = pd.read_csv('data/raw_data/bangumi15M/raw_data/Subjects.csv', index_col=0)

items = []
for sid in q_scaled.index:
    meta = subj.loc[int(sid)] if int(sid) in subj.index else None
    items.append({
        'id': int(sid),
        'name': str(meta['name']) if meta is not None and pd.notna(meta.get('name')) else '',
        'name_cn': str(meta['name_cn']) if meta is not None and pd.notna(meta.get('name_cn')) else '',
        'date': str(meta['date']) if meta is not None and pd.notna(meta.get('date')) else '',
        'url': f'https://bgm.tv/subject/{sid}',
        'score': round(float(meta['score']), 1) if meta is not None and pd.notna(meta.get('score')) else None,
        'hb_score': round(float(q_scaled[sid]), 2),
    })

for i, it in enumerate(items):
    sid_str = str(it['id'])
    it['hb_rank'] = i + 1
    if sid_str in r_imdb.index:
        it['imdb_score'] = round(float(r_imdb[sid_str]), 2)
        it['imdb_rank'] = list(r_imdb.index).index(sid_str) + 1
    else:
        it['imdb_score'] = None; it['imdb_rank'] = None
    it['delta'] = (it['imdb_rank'] or 0) - it['hb_rank']

with open('web/data/rankings.json', 'w', encoding='utf-8') as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

params = hb.guide.median(hb.svi_result.params)
stats = {
    'model': 'HierarchicalBayes + precision-weighted prior',
    'n_ref': int(n_ref),
    'n_ref_note': '评分人数中位数，纯数据驱动，零人工参数。低评分条目先验自动收紧。',
    'mu_q': float(params['mu_q']),
    'tau_q': float(params['tau_q']),
    'tau_b': float(params['tau_b']),
    'n_users': int(d['N_users']), 'n_subjects': int(d['N_subjects']),
    'n_ratings': int(len(d['obs'])),
    'user_bias_median': float(hb.user_bias_.median()),
    'user_noise_median': float(hb.user_noise_.median()),
    'loss_final': float(losses[-1]),
    'runtime_s': int(elapsed), 'tau_vs_imdb': round(tau, 4),
}
with open('web/data/stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

hb.save('solution')

print(f'\nGenerated {len(items)} items')
print('Top 10:')
for it in items[:10]:
    d = it['delta']
    ds = f'+{d}' if d > 0 else str(d)
    print(f'  {it["hb_rank"]:2d}. {it["name_cn"] or it["name"]} ({it["hb_score"]}) IMDb#{it["imdb_rank"]} Δ={ds}')

# 检查用户关心的条目
print('\n=== 用户关心的条目 ===')
for sid in [37183, 211269, 67753, 23304]:
    for it in items:
        if it['id'] == sid:
            print(f'{sid}: HB#{it["hb_rank"]} IMDb#{it["imdb_rank"]} Δ={it["delta"]}')
