import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
from hbayes.model import HierarchicalBayesRank
from fusion.baselines import imdb_bayesian_average
from fusion.evaluate import kendall_tau, _to_str_index

base = '.'
csv = f'{base}/data/_subject_user_matrix.csv'

hb = HierarchicalBayesRank(csv, rng_seed=42)
d = hb.data
n_sub = d['N_subjects']
n_rat = len(d['obs'])
print(f'n={n_sub} users={d["N_users"]} ratings={n_rat:,}')

t0 = time.time()
hb.fit(steps=10000, lr=0.001, verbose=False)
elapsed = time.time()-t0
losses = hb.svi_result.losses
cp = np.array([float(x) for x in hb.cutpoints_])
ok = not any(np.isnan(x) for x in cp)
print(f'{elapsed:.0f}s loss={losses[-1]:.0f} cp_ok={ok}')
if ok:
    print(f'cp={np.round(cp,1)}')
    r_hb = _to_str_index(hb.rank())
    r_imdb = _to_str_index(imdb_bayesian_average(csv))
    tau, _ = kendall_tau(r_hb, r_imdb)
    print(f'tau={tau:.4f}')
    for i,(sid,s) in enumerate(hb.rank(top_n=10).items()):
        print(f'  {i+1}. {sid}: {s:.4f}')
