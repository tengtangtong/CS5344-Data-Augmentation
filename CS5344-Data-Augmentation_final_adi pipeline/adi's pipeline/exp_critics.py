"""Experiment: other models in the trimming critic (Pipeline 3 machinery).

The critic ranks the over-generated candidates; the least attack-like 10% per
category tuple are kept. Components (rank-averaged, as Pipeline 1 does):
  clf  HistGB P(anomaly)                 (current UNSW critic)
  ae   autoencoder reconstruction error, trained on REAL NORMALS only (unsupervised)
  mlp  MLP classifier P(anomaly)          (supervised, smooth decision surface)
  lr   logistic regression P(anomaly)     (supervised, linear)
Same generator pool per split; only the critic used for trimming changes. The
learned UNSW shaping plan is applied; seen and pseudo-novel attack types are
scored with exact PyOD ECOD + IForest.

Usage: python exp_critics.py UNSW-NB15 0 1 2
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import QuantileTransformer

import pipeline3 as P

name = sys.argv[1] if len(sys.argv) > 1 else "UNSW-NB15"
rots = [int(r) for r in sys.argv[2:]] or [0, 1, 2]
cfg = next(c for c in P.DATASETS if c["name"] == name)
cats = cfg["cats"]
S = P.settings_for(name)
df, feats, nums = P.load_data(cfg)
plan = {"UNSW-NB15": {'spkts': -1.0, 'dpkts': 50.0, 'sbytes': 50.0, 'dbytes': -1.0, 'sloss': 50.0, 'dloss': 4.0,
                      'response_body_len': -1.0, 'ct_src_dport_ltm': 50.0, 'dtcpb': 4.0, 'ct_dst_src_ltm': 4.0},
        "NSL-KDD": {}}[name]
BASE_SCORER = S["scorer"]                       # "clf" (UNSW) or "clf+real" (NSL)
VARIANTS = [BASE_SCORER, BASE_SCORER + "+ae", "ae", BASE_SCORER + "+mlp", BASE_SCORER + "+lr", BASE_SCORER + "+mlp+ae"]


class Features:
    """Gaussian-quantile numerics + one-hot categoricals, fitted on real normals."""

    def __init__(self, normal):
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(normal[nums])
        self.levels = {c: np.array(sorted(normal[c].astype(str).unique())) for c in cats}

    def __call__(self, d):
        parts = [np.clip(self.qt.transform(d[nums]), -5, 5)]
        for c in cats:
            parts.append((d[c].astype(str).to_numpy()[:, None] == self.levels[c][None]).astype(float))
        return np.hstack(parts)


def chunked(fn, X, size=200000):
    return np.concatenate([fn(X[i:i + size]) for i in range(0, len(X), size)])


rows = []
for r in rots:
    t0 = time.time()
    sp = P.Split(df, feats, nums, cats, name, S, P.GENERATOR[name], cfg["seed"] + 7 + 313 * r, rotation=r)
    ctx, normal = sp.ctx, sp.ctx["normal"]
    abnormal = sp.train[sp.train[P.LABEL] == 1][feats].reset_index(drop=True)
    F = Features(normal)
    Xn, Xa, Xp = F(normal), F(abnormal), F(ctx["pool"])
    # autoencoder on real normals (bottleneck 8)
    ae = MLPRegressor(hidden_layer_sizes=(64, 8, 64), max_iter=60, early_stopping=True, random_state=r).fit(Xn, Xn)
    ctx["components"]["ae"] = chunked(lambda X: ((ae.predict(X) - X) ** 2).mean(1), Xp)
    # supervised partners (normal 0 vs seen anomaly 1)
    Xs, ys = np.vstack([Xn, Xa]), np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=r).fit(Xs, ys)
    ctx["components"]["mlp"] = chunked(lambda X: mlp.predict_proba(X)[:, 1], Xp)
    lr = LogisticRegression(max_iter=500, C=1.0).fit(Xs, ys)
    ctx["components"]["lr"] = chunked(lambda X: lr.predict_proba(X)[:, 1], Xp)
    for scorer in VARIANTS:
        final, _, fb = P.select_indices(ctx, dict(S, scorer=scorer), P.P3["learn_n"], sp.seed)
        syn = P.assemble(ctx, final, fb, plan)
        A = sp.enc(syn)
        e, f = P.ecod_scores(A, sp.Xval), P.iforest_scores(A, sp.Xval, *P.P3["eval_iforest"])
        o = dict(rotation=r, critic=scorer)
        for tag, m in [("seen", sp.m_seen), ("novel", sp.m_unseen)]:
            o[tag + "_E"] = P.average_precision_score(sp.y[m], e[m])
            o[tag + "_I"] = P.average_precision_score(sp.y[m], f[m])
            o[tag] = (o[tag + "_E"] + o[tag + "_I"]) / 2
        rows.append(o)
        print("{} {:<16} seen {:.4f}  novel {:.4f}  (ECOD novel {:.4f}, IForest novel {:.4f})".format(
            r, scorer, o["seen"], o["novel"], o["novel_E"], o["novel_I"]), flush=True)
    print("rotation {} done in {:.1f} min".format(r, (time.time() - t0) / 60), flush=True)
    del sp, ctx

tab = pd.DataFrame(rows)
tab.to_csv("exp_critics_{}.csv".format(name), index=False)
b = tab[tab.critic == BASE_SCORER].set_index("rotation")
d = tab.set_index("rotation").copy()
cols = ["seen", "novel", "novel_E", "novel_I"]
for c in cols:
    d[c] = d[c] - b[c].reindex(d.index).to_numpy()
print("\nΔ vs current critic (mean / worst rotation):")
print(d.groupby("critic")[cols].agg(["mean", "min"]).round(4).to_string())
