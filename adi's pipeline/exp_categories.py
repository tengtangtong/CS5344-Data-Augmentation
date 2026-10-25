"""Option 3: category-frequency shaping, tested on Pipeline 17 behaviour splits.

Every variant keeps the same numeric rows and only changes HOW OFTEN each
category / category tuple appears in the synthetic set (rows are dropped and
replaced by re-drawing rows of other tuples). Scored on seen and pseudo-novel
attack types, under BOTH plausible organiser encodings (one-hot and ordinal).
Not a submission script.

Usage: python exp_categories.py UNSW-NB15 0 1 2
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

import pipeline17 as p17
from pipeline16 import DATASETS, LABEL, ecod_scores, iforest_scores, load_data, settings_for, tuple_key

name = sys.argv[1] if len(sys.argv) > 1 else "UNSW-NB15"
rots = [int(r) for r in sys.argv[2:]] or [0, 1, 2]
cfg = next(c for c in DATASETS if c["name"] == name)
cats = cfg["cats"]
S = settings_for(name)
df, feats, nums = load_data(cfg)
plan17 = {"UNSW-NB15": {'spkts': 50.0, 'dpkts': 50.0, 'sbytes': 50.0, 'dbytes': 4.0, 'sloss': 50.0, 'dloss': 4.0,
                        'response_body_len': 50.0, 'ct_src_dport_ltm': 50.0},
          "NSL-KDD": {'duration': 4.0}}[name]


class Ordinal:
    """Label codes: levels sorted, unseen levels → −1 (as pipeline16.encode)."""

    def __init__(self, syn):
        self.levels = {c: {v: i for i, v in enumerate(sorted(syn[c].astype(str).unique()))} for c in cats}

    def __call__(self, d):
        parts = [d[nums].to_numpy(float)]
        for c in cats:
            parts.append(d[c].astype(str).map(self.levels[c]).fillna(-1).to_numpy(float)[:, None])
        return np.hstack(parts)


def reshape(syn, keep_mask, rng):
    """Drop rows not in keep_mask and refill to the same size by re-drawing kept rows."""
    kept = syn[keep_mask]
    if len(kept) == len(syn):
        return syn
    extra = kept.iloc[rng.integers(0, len(kept), len(syn) - len(kept))]
    return pd.concat([kept, extra], ignore_index=True)


def reweight(syn, w_row, rng):
    """Resample rows with probability ∝ w_row (category-level tilt)."""
    p = w_row / w_row.sum()
    return syn.iloc[rng.choice(len(syn), len(syn), replace=True, p=p)].reset_index(drop=True)


def attack_rates(train, col_or_tuple):
    """Smoothed P(attack | level) on the TRAINING side (seen types only)."""
    key = tuple_key(train, cats) if col_or_tuple == "tuple" else train[col_or_tuple].astype(str).to_numpy()
    y = train[LABEL].to_numpy()
    t = pd.DataFrame(dict(k=key, y=y)).groupby("k")["y"].agg(["sum", "count"])
    prior = y.mean()
    return ((t["sum"] + 5 * prior) / (t["count"] + 5)).to_dict()


def score(sp, syn):
    out = {}
    for enc_name, enc in [("onehot", p17.FixedEncoder(syn, nums, cats)), ("ordinal", Ordinal(syn))]:
        A, B = enc(syn), enc(sp.val)
        e, f = ecod_scores(A, B), iforest_scores(A, B, *p17.P17["eval_iforest"])
        for tag, m in [("seen", sp.m_seen), ("novel", sp.m_unseen)]:
            out["{}_{}".format(enc_name, tag)] = (ap(sp.y[m], e[m]) + ap(sp.y[m], f[m])) / 2
    out["seen"] = (out["onehot_seen"] + out["ordinal_seen"]) / 2
    out["novel"] = (out["onehot_novel"] + out["ordinal_novel"]) / 2
    return out


rows = []
for r in rots:
    t0 = time.time()
    sp = p17.Split(df, feats, nums, cats, name, S, p17.GENERATOR[name], cfg["seed"] + 7 + 313 * r, rotation=r)
    rng = np.random.default_rng(r)
    base = p17.assemble(sp.ctx, sp.final, sp.fb, plan17)
    variants = [("base (P17)", base)]
    # A. drop single categories that are mostly attacks on the training side
    for t in [0.9, 0.75, 0.6]:
        keep = np.ones(len(base), bool)
        dropped = []
        for c in cats:
            ar = attack_rates(sp.train, c)
            bad = [v for v, a in ar.items() if a > t]
            dropped += ["{}={}".format(c, v) for v in bad if (base[c].astype(str) == v).any()]
            keep &= ~base[c].astype(str).isin(bad).to_numpy()
        variants.append(("drop categories P(attack)>{}".format(t), reshape(base, keep, rng)))
        print("   drop>{}: {} levels, {:.1%} of rows: {}".format(t, len(dropped), 1 - keep.mean(), dropped[:12]), flush=True)
    # B. drop whole tuples that are mostly attacks
    ar_t = attack_rates(sp.train, "tuple")
    key = tuple_key(base, cats)
    for t in [0.75, 0.5]:
        keep = np.array([ar_t.get(k, 0.0) <= t for k in key])
        variants.append(("drop tuples P(attack)>{}".format(t), reshape(base, keep, rng)))
    # C. tilt tuple frequencies by (1 − P(attack | tuple))^g
    pa = np.array([ar_t.get(k, 0.0) for k in key])
    for g in [2.0, 8.0]:
        variants.append(("tilt tuples (1-P)^{:g}".format(g), reweight(base, (1 - pa) ** g + 1e-9, rng)))
    # D. drop rare tuples (< 0.1% of synthetic rows) — makes rare combinations isolated
    share = pd.Series(key).map(pd.Series(key).value_counts(normalize=True)).to_numpy()
    variants.append(("drop rare tuples <0.1%", reshape(base, share >= 0.001, rng)))
    # E. diagnostic: flatten tuple frequencies (every tuple equally common)
    variants.append(("flatten tuples (diagnostic)", reweight(base, 1.0 / share, rng)))
    for tag, syn in variants:
        rows.append(dict(rotation=r, variant=tag, **score(sp, syn)))
        print("{} {:<32} seen {:.4f} novel {:.4f} | onehot {:.4f}/{:.4f} ordinal {:.4f}/{:.4f}".format(
            r, tag, rows[-1]["seen"], rows[-1]["novel"], rows[-1]["onehot_seen"], rows[-1]["onehot_novel"],
            rows[-1]["ordinal_seen"], rows[-1]["ordinal_novel"]), flush=True)
    print("rotation {} done in {:.1f} min".format(r, (time.time() - t0) / 60), flush=True)

tab = pd.DataFrame(rows)
tab.to_csv("exp_categories_{}.csv".format(name), index=False)
base = tab[tab.variant == "base (P17)"].set_index("rotation")
cols = ["seen", "novel", "onehot_seen", "onehot_novel", "ordinal_seen", "ordinal_novel"]
delta = tab.set_index("rotation")
for c in cols:
    delta[c] = delta[c] - base[c].reindex(delta.index).to_numpy()
summ = delta.groupby("variant")[cols].agg(["mean", "min"])
print("\nΔ vs base (mean / worst rotation):")
print(summ.round(4).to_string())
