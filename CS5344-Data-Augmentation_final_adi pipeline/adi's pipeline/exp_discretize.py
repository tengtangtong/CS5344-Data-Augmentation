"""Experiment: does discretising the synthetic numeric values help ECOD / IForest?

Post-hoc variants of the final Pipeline 3 files, scored in-sample on the released
training rows (exact PyOD ECOD, one-hot; IForest 3 seeds x 200 trees). In-sample
numbers are optimistic in level, but tracked Kaggle at Spearman +0.83 on UNSW.
  base      : file as submitted (integer columns already rounded)
  snap      : every numeric value snapped to the nearest value seen in real normals
  qbin_B    : real-normal quantile bins (B per column), value -> real-normal bin median

Usage: python pipeline3.py; python exp_discretize.py [NSL-KDD] [UNSW-NB15]
       (reads pipeline3_output/ or the folder in P3_OUT)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from pipeline1 import DATASETS, encode, iforest_scores
from scipy.stats import skew
from pipeline1 import _ecdf_cols


def ecod(Xtr, Xte):
    X = np.vstack([Xtr, Xte]).astype(float)
    U_l, U_r = -np.log(_ecdf_cols(X)), -np.log(_ecdf_cols(-X))
    with np.errstate(all="ignore"):
        sk = np.nan_to_num(np.sign(skew(X, axis=0)))
    U_skew = U_l * -1 * np.sign(sk - 1) + U_r * np.sign(sk + 1)
    return np.maximum(U_skew, np.maximum(U_l, U_r)).sum(1)[-len(Xte):]


HERE = Path(__file__).resolve().parent
OUT = Path(__import__("os").environ.get("P3_OUT", "pipeline3_output"))
FILES = {n: OUT / "submission_{}_p3.csv".format(n) for n in ("NSL-KDD", "UNSW-NB15")}


def find(fname):
    for p in [Path(fname), HERE / fname, HERE.parent / fname, HERE.parent / "data" / fname]:
        if p.exists():
            return p
    raise FileNotFoundError(fname)


def snap(v, support):
    i = np.clip(np.searchsorted(support, v), 1, len(support) - 1)
    lo, hi = support[i - 1], support[i]
    return np.where(np.abs(v - lo) <= np.abs(hi - v), lo, hi)


def qbin(v, ref, B):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, B + 1)))
    if len(edges) < 3:
        return v
    bi = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    rb = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, len(edges) - 2)
    med = np.array([np.median(ref[rb == k]) if (rb == k).any() else edges[k] for k in range(len(edges) - 1)])
    return med[bi]


for name in sys.argv[1:] or list(FILES):
    cfg = next(c for c in DATASETS if c["name"] == name)
    cats = cfg["cats"]
    test = pd.read_csv(find("train_{}.csv".format(name)))
    for c in cats:
        test[c] = test[c].astype(str)
    y = test.pop("is_anomaly").to_numpy()
    real = test[y == 0]
    syn0 = pd.read_csv(FILES[name]).drop(columns=["id"], errors="ignore")
    for c in cats:
        syn0[c] = syn0[c].astype(str)
    nums = [c for c in test.columns if c not in cats]
    variants = {"base": lambda d: d}
    def _snap(d):
        d = d.copy()
        for c in nums:
            d[c] = snap(d[c].to_numpy(float), np.unique(real[c].to_numpy(float)))
        return d
    variants["snap"] = _snap
    for B in [50, 20, 10, 5]:
        def _q(d, B=B):
            d = d.copy()
            for c in nums:
                d[c] = qbin(d[c].to_numpy(float), real[c].to_numpy(float), B)
            return d
        variants["qbin_{}".format(B)] = _q
    for tag, fn in variants.items():
        syn = fn(syn0)
        A, Bm = encode(syn, test, nums, cats, "onehot")
        e, f = ap(y, ecod(A, Bm)), ap(y, iforest_scores(A, Bm, 3, 200))
        print("{:<10} {:<8} ECOD {:.4f}  IForest {:.4f}  mean {:.4f}  dup rows {:.1%}".format(
            name, tag, e, f, (e + f) / 2, syn.duplicated().mean()), flush=True)
