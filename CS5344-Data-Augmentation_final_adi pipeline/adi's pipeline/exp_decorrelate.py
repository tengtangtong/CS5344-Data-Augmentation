"""Experiment: decorrelate synthetic feature columns, scored on Pipeline 2
behaviour splits (seen vs pseudo-novel attack types), exact PyOD ECOD + IForest.

Decorrelation = independently permuting a column's values across rows. Each
column's marginal is unchanged, so ECOD (per-column ECDFs) should not move except
through recomputed derived columns; only IForest sees the joint structure.

Usage: python exp_decorrelate.py UNSW-NB15 0 1 2
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score

import pipeline2 as p2
from pipeline1 import DATASETS, LABEL, load_data, settings_for, tuple_key

name = sys.argv[1] if len(sys.argv) > 1 else "UNSW-NB15"
rots = [int(r) for r in sys.argv[2:]] or [0, 1, 2]
cfg = next(c for c in DATASETS if c["name"] == name)
cats = cfg["cats"]
S = settings_for(name)
df, feats, nums = load_data(cfg)
plan = {"UNSW-NB15": {'spkts': -1.0, 'dpkts': 50.0, 'sbytes': 50.0, 'dbytes': -1.0, 'sloss': 50.0, 'dloss': 4.0,
                      'response_body_len': -1.0, 'ct_src_dport_ltm': 50.0, 'dtcpb': 4.0, 'ct_dst_src_ltm': 4.0},
        "NSL-KDD": {}}[name]


def permute(syn, spec, cols, rng, within_tuple):
    out = syn.copy()
    key = tuple_key(out, cats) if within_tuple else np.zeros(len(out), dtype=int)
    for c in cols:
        v = out[c].to_numpy().copy()
        for k in pd.unique(key):
            idx = np.where(key == k)[0]
            v[idx] = v[idx][rng.permutation(len(idx))]
        out[c] = v
    return p2.p1.restore_derived(out, spec)[spec["feats"]]


def score(sp, syn):
    A = sp.enc(syn)
    e, f = p2.ecod_scores(A, sp.Xval), p2.iforest_scores(A, sp.Xval, *p2.P2["eval_iforest"])
    o = {}
    for tag, m in [("seen", sp.m_seen), ("novel", sp.m_unseen)]:
        o[tag + "_E"], o[tag + "_I"] = ap(sp.y[m], e[m]), ap(sp.y[m], f[m])
        o[tag] = (o[tag + "_E"] + o[tag + "_I"]) / 2
    return o


rows = []
for r in rots:
    t0 = time.time()
    sp = p2.Split(df, feats, nums, cats, name, S, p2.GENERATOR[name], cfg["seed"] + 7 + 313 * r, rotation=r)
    spec = sp.ctx["spec"]
    rng = np.random.default_rng(100 + r)
    base = p2.assemble(sp.ctx, sp.final, sp.fb, plan)
    y = sp.train[LABEL].to_numpy()
    gen = [c for c in spec["gen_nums"] if sp.ctx["normal"][c].nunique() > 1]
    auc = {c: abs(roc_auc_score(y, sp.train[c].to_numpy(float)) - 0.5) for c in gen}
    weak = [c for c in gen if auc[c] < 0.10]
    strong = [c for c in gen if auc[c] >= 0.10]
    variants = [("base", base),
                ("permute all, within tuple", permute(base, spec, gen, rng, True)),
                ("permute all, global", permute(base, spec, gen, rng, False)),
                ("permute weak only (|AUC-.5|<.1), within tuple", permute(base, spec, weak, rng, True)),
                ("permute strong only, within tuple", permute(base, spec, strong, rng, True))]
    for tag, syn in variants:
        rows.append(dict(rotation=r, variant=tag, **score(sp, syn)))
        x = rows[-1]
        print("{} {:<46} seen {:.4f} (E {:.4f} I {:.4f})  novel {:.4f} (E {:.4f} I {:.4f})".format(
            r, tag, x["seen"], x["seen_E"], x["seen_I"], x["novel"], x["novel_E"], x["novel_I"]), flush=True)
    print("rotation {}: {} weak / {} strong columns, {:.1f} min".format(r, len(weak), len(strong), (time.time() - t0) / 60), flush=True)

tab = pd.DataFrame(rows)
tab.to_csv("exp_decorrelate_{}.csv".format(name), index=False)
b = tab[tab.variant == "base"].set_index("rotation")
d = tab.set_index("rotation").copy()
cols = ["seen", "novel", "seen_E", "seen_I", "novel_E", "novel_I"]
for c in cols:
    d[c] = d[c] - b[c].reindex(d.index).to_numpy()
print("\nΔ vs base (mean / worst rotation):")
print(d.groupby("variant")[cols].agg(["mean", "min"]).round(4).to_string())
