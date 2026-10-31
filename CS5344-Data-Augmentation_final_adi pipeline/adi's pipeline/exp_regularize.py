"""Experiment: regularise the HistGB anomaly classifier (the trimming critic).

Each variant refits the whole Pipeline 2 training side (critic → weights → VAE →
over-generate → trim → sculpt) on a behaviour split, applies the learned UNSW-NB15
shaping plan, and scores seen and pseudo-novel attack types with the exact ECOD.

Usage: python exp_regularize.py UNSW-NB15 0 1 2
"""
import sys
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import pipeline2 as p2
import pipeline1 as p1
from pipeline1 import DATASETS, load_data, settings_for

name = sys.argv[1] if len(sys.argv) > 1 else "UNSW-NB15"
rots = [int(r) for r in sys.argv[2:]] or [0, 1, 2]
cfg = next(c for c in DATASETS if c["name"] == name)
S = settings_for(name)
df, feats, nums = load_data(cfg)
plan = {"UNSW-NB15": {'spkts': -1.0, 'dpkts': 50.0, 'sbytes': 50.0, 'dbytes': -1.0, 'sloss': 50.0, 'dloss': 4.0,
                      'response_body_len': -1.0, 'ct_src_dport_ltm': 50.0, 'dtcpb': 4.0, 'ct_dst_src_ltm': 4.0},
        "NSL-KDD": {}}[name]

VARIANTS = {                                   # HistGB keyword overrides
    "default (current)": {},
    "L2 = 1, leaves ≥ 100": dict(l2_regularization=1.0, min_samples_leaf=100),
    "shallow: depth 3, 150 iters": dict(max_depth=3, max_iter=150),
    "strong: depth 2, L2 = 10, leaves ≥ 200": dict(max_depth=2, l2_regularization=10.0, min_samples_leaf=200),
    "early stopping (10% val)": dict(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
}
current = {}


def _clf(self, seed):
    kw = dict(max_iter=300, learning_rate=0.08, random_state=seed, categorical_features=self.enc.mask)
    kw.update(current)
    return HistGradientBoostingClassifier(**kw)


p1.AnomalyModel._clf = _clf            # BehaviourAnomalyModel inherits it

rows = []
for r in rots:
    for tag, over in VARIANTS.items():
        t0 = time.time()
        current.clear()
        current.update(over)
        sp = p2.Split(df, feats, nums, cfg["cats"], name, S, p2.GENERATOR[name], cfg["seed"] + 7 + 313 * r, rotation=r)
        m = sp.score(plan, 0.5, 0.0)
        rows.append(dict(rotation=r, variant=tag, seen=m["seen"], novel=m["unseen"],
                         seen_E=m["seen_ecod"], seen_I=m["seen_iforest"], novel_E=m["unseen_ecod"], novel_I=m["unseen_iforest"]))
        print("{} {:<40} seen {:.4f}  novel {:.4f}  ({:.1f} min)".format(r, tag, m["seen"], m["unseen"], (time.time() - t0) / 60), flush=True)
        del sp

tab = pd.DataFrame(rows)
tab.to_csv("exp_regularize_{}.csv".format(name), index=False)
b = tab[tab.variant == "default (current)"].set_index("rotation")
d = tab.set_index("rotation").copy()
cols = ["seen", "novel", "novel_E", "novel_I"]
for c in cols:
    d[c] = d[c] - b[c].reindex(d.index).to_numpy()
print("\nΔ vs current critic (mean / worst rotation):")
print(d.groupby("variant")[cols].agg(["mean", "min"]).round(4).to_string())
