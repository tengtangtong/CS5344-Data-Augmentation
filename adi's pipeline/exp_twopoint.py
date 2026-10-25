"""Experiment: two-point neutralisation of weak columns, scored on Pipeline 17
behaviour splits (seen vs pseudo-novel types). Not part of any submission."""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pipeline17 as p17
from pipeline16 import DATASETS, load_data, settings_for, LABEL

name = sys.argv[1] if len(sys.argv) > 1 else "UNSW-NB15"
rots = [int(r) for r in sys.argv[2:]] or [0, 2]
cfg = next(c for c in DATASETS if c["name"] == name)
S = settings_for(name)
df, feats, nums = load_data(cfg)
plan17 = {"UNSW-NB15": {'spkts': 50.0, 'dpkts': 50.0, 'sbytes': 50.0, 'dbytes': 4.0, 'sloss': 50.0, 'dloss': 4.0,
                        'response_body_len': 50.0, 'ct_src_dport_ltm': 50.0},
          "NSL-KDD": {'duration': 4.0}}[name]


def two_point(syn, spec, cols, p_lo=0.5):
    """Rank-preserving: lowest p_lo of rows → real-normal min, the rest → real-normal max."""
    out = syn.copy()
    n = len(out)
    for c in cols:
        r = np.argsort(np.argsort(out[c].to_numpy(float), kind="stable"), kind="stable")
        lo, hi = spec["ranges"][c]
        out[c] = np.where(r < p_lo * n, lo, hi)
    return p17.p16.restore_derived(out, spec)[spec["feats"]]


def score(sp, syn):
    A = sp.enc(syn)
    e = p17.ecod_scores(A, sp.Xval)
    f = p17.iforest_scores(A, sp.Xval, *p17.P17["eval_iforest"])
    from sklearn.metrics import average_precision_score as ap
    o = {}
    for tag, m in [("seen", sp.m_seen), ("novel", sp.m_unseen)]:
        o[tag + "_E"], o[tag + "_I"] = ap(sp.y[m], e[m]), ap(sp.y[m], f[m])
        o[tag] = (o[tag + "_E"] + o[tag + "_I"]) / 2
    return o


rows = []
for r in rots:
    t0 = time.time()
    sp = p17.Split(df, feats, nums, cfg["cats"], name, S, p17.GENERATOR[name], cfg["seed"] + 7 + 313 * r, rotation=r)
    spec = sp.ctx["spec"]
    tr = sp.train
    y = tr[LABEL].to_numpy()
    auc = {c: abs(roc_auc_score(y, tr[c].to_numpy(float)) - 0.5) if tr[c].nunique() > 1 else 0.0 for c in spec["gen_nums"]}
    order = sorted(auc, key=lambda c: -auc[c])                     # most informative first
    base = p17.assemble(sp.ctx, sp.final, sp.fb, plan17)
    variants = [("P17 plan", base)]
    for keep in [4, 8, 12, 16]:
        weak = [c for c in order[keep:] if c not in spec["integer"] or sp.ctx["normal"][c].nunique() > 2]
        variants.append(("two-point all but top {}".format(keep), two_point(base, spec, weak)))
    for d in [0.05, 0.10, 0.15]:
        weak = [c for c in order if auc[c] < d]
        variants.append(("two-point |AUC-.5|<{}".format(d), two_point(base, spec, weak)))
    for tag, syn in variants:
        rows.append(dict(rotation=r, variant=tag, **score(sp, syn)))
        print(r, tag, {k: round(v, 4) for k, v in rows[-1].items() if k not in ("rotation", "variant")}, flush=True)
    print("top informative:", order[:16], "{:.1f} min".format((time.time() - t0) / 60), flush=True)
tab = pd.DataFrame(rows)
print(tab.groupby("variant")[["seen", "novel", "seen_E", "seen_I", "novel_E", "novel_I"]].mean().round(4).sort_values("novel"))
tab.to_csv("exp_twopoint_{}.csv".format(name), index=False)
