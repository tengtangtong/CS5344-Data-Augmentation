# =====================================================================
# PIPELINE 17 — Pipeline 16 + three detector-aware learning steps
# Normal-only synthetic data for fixed ECOD + IForest detectors
#
# Builds on pipeline16.py (imported, not copied: every Pipeline 16 stage —
# curation, derived flow, anomaly model, conditional VAE, over-generate,
# trim, sculpt, validation — is reused unchanged). New in Pipeline 17:
#
#   A. Behaviour-based pseudo-novel types
#      Anomalies are described by WHAT THEY DO, not by all 40 raw columns:
#      traffic volume, traffic direction (source vs destination side),
#      timing, connection-rate counters, error/loss counters, TTL/window,
#      protocol and connection state. Column groups come from generic
#      name patterns that match both datasets (no per-dataset column lists).
#      Clusters in that space are pseudo attack types; the ones FARTHEST
#      from the rest are held out as "pseudo-novel" — a pessimistic stand-in
#      for the unseen A_val / A_test types. Used for
#        • the validation split that A and C learn on, and
#        • the leave-one-type-out anomaly model (UNSW), replacing the
#          Pipeline 16 clustering on raw features.
#
#   B. Joint subset selection for BOTH detectors
#      Two cheap per-candidate surrogates mimic how each detector is hurt
#      when a synthetic row sits where attacks live:
#        ECOD    → per-column naive-Bayes log-likelihood ratio (ECOD sums
#                  independent per-column tail evidence; so does this)
#        IForest → ExtraTrees with one random feature per split (random
#                  axis-aligned splits, like isolation trees)
#      Harm h = α·rank(ECOD harm) + (1−α)·rank(IForest harm). Within every
#      category tuple, the ρ most harmful selected rows are swapped for the
#      least harmful unused trimmed candidates. (α, ρ) are chosen by the real
#      detectors' AUPRC on the behaviour validation split — ρ = 0 is
#      Pipeline 16, so the search can always fall back to it.
#
#   C. Learned per-column spreading κ with an unseen-type penalty
#      Pipeline 16 spreads one fixed column set with one κ. Pipeline 17 runs
#      coordinate ascent over κ_j ∈ {off, 4, 50} for every weak, heavy-tailed,
#      unbounded column, starting from the Pipeline 16 plan. Objective:
#        J = ½ (S_seen + S_unseen) − λ · max(0, ΔS_seen − ΔS_unseen)
#      and a change is accepted only if it raises J by > ε AND does not
#      lower S_unseen by more than a tolerance. Gains that come only from
#      seen attack types are penalised, which is what hurt us on Kaggle.
#
# Learning protocol (no leakage): A/B/C learn on behaviour splits of the
# RELEASED training data only. On each split the critic, the surrogates and
# the generator see only the training side; pseudo-novel types are never on
# it. The learned plan, α and ρ are then REPLAYED on a fit to all released
# rows. Validation/test/Kaggle rows are never touched.
#
# MODE = "final"    → learn on N_LEARN_SPLITS behaviour splits, then write
#                     submission_<dataset>_p17.csv + metadata JSON
# MODE = "evaluate" → learn on N_LEARN_SPLITS splits, then compare Pipeline 16
#                     vs Pipeline 17 on an INDEPENDENT behaviour split whose
#                     pseudo-novel types differ from the learning splits'
#                     (seen / pseudo-novel / overall AUPRC). Use for the report.
# MODE = "final+evaluate" → both, sharing one learning pass
#
# Requires: pipeline16.py in the same folder; numpy pandas scipy scikit-learn
# =====================================================================

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

import pipeline16 as p16
from pipeline16 import (LABEL, DATASETS, MIN_CANDIDATES, CodeEncoder, density_ratio,
                        ecod_scores, iforest_scores, load_data, make_quotas, postprocess, prepare,
                        save_submission, settings_for, spread_columns, tuple_key, validate,
                        weighted_choice_noreplace, weights_for)

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
MODE = os.environ.get("P17_MODE", "final")   # "final" | "evaluate" | "final+evaluate"
OUTDIR = Path(os.environ.get("P17_OUT", "pipeline17_output"))
GENERATOR = {"NSL-KDD": "vae", "UNSW-NB15": "vae"}   # Pipeline 17 needs a trimmed generator (not the baseline)
ECOD_RULE = os.environ.get("P17_ECOD", "pyod")   # "pyod" (exact PyOD rule) | "old" (Pipeline 16 fallback)
FAST = os.environ.get("P17_FAST") == "1"   # quick smoke test: P17_FAST=1 python pipeline17.py

P17 = dict(
    behaviour_loto=True,            # A: leave-one-type-out critic uses behaviour types
    learn_kappa=True,               # C: learned per-column κ
    joint_select=True,              # B: joint subset selection for ECOD + IForest
    n_learn_splits=1 if FAST else 2,
    learn_n=4000 if FAST else 20000,            # synthetic rows per learning run
    hidden_share=0.25,              # target share of anomalies held out as pseudo-novel
    behaviour_k=(4, 10),            # number of behaviour types searched (silhouette)
    val_normal_frac=0.25,           # real normals moved to the validation side
    val_seen_frac=0.25,             # seen-type anomalies moved to the validation side
    kappa_options=(0.0, 4.0, 50.0, -1.0),  # 0 = column not spread, -1 = two-point (half min, half max)
    kappa_delta_max=0.25,           # candidate columns: |AUC − 0.5| < this (P16 used 0.10)
    kappa_passes=1,
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    rhos=(0.0, 0.1, 0.2, 0.35, 0.5),
    lam=1.0,                        # unseen-type penalty weight λ
    eps=0.002,                      # minimum J gain to accept a change
    unseen_tol=0.002,               # max allowed drop in pseudo-novel AUPRC
    eval_iforest=(2, 100) if FAST else (3, 150),  # (seeds, trees) of the proxy IForest
    et_trees=60 if FAST else 150,   # ExtraTrees IForest-surrogate size
    nb_bins=20,                     # ECOD-surrogate bins per column
)

# Behaviour groups: generic column-name patterns, first match wins.
BEHAVIOUR_GROUPS = [
    ("error",  r"err|loss|wrong|urgent|fail|hot|compromis|shell|root|file|access|guest"),
    ("timing", r"dur|time|jit|inpkt|rtt|synack|ackdat"),
    ("ttl",    r"ttl|win|tcpb"),
    ("volume", r"byte|pkt|load|len|mean|depth"),
    ("rate",   r"count|ct_|rate|srv|host"),
]
SRC_SIDE, DST_SIDE = r"^(s(?!rv|ame|u_)|src_)", r"^(d(?!iff|ur)|dst_)"

if FAST:   # CVAE binds its epoch default at import time, so patch the default itself
    _d = list(p16.CVAE.__init__.__defaults__)
    _d[5] = 8
    p16.CVAE.__init__.__defaults__ = tuple(_d)


# =====================================================================
# 2. A — BEHAVIOUR-BASED PSEUDO-NOVEL TYPES
# =====================================================================
def behaviour_groups(nums):
    groups = {}
    for c in nums:
        g = next((g for g, pat in BEHAVIOUR_GROUPS if re.search(pat, c)), None)
        if g:
            groups.setdefault(g, []).append(c)
    vol = groups.get("volume", [])
    src = [c for c in vol if re.search(SRC_SIDE, c)]
    dst = [c for c in vol if re.search(DST_SIDE, c)]
    return groups, src, dst


class BehaviourSpace:
    """Rows → low-dimensional behaviour descriptor:
    mean quantile rank per behaviour group, source-minus-destination volume
    (direction), and down-weighted one-hots of the categorical columns."""

    def fit(self, rows, nums, cats):
        self.groups, self.src, self.dst = behaviour_groups(nums)
        self.cols = sorted({c for cs in self.groups.values() for c in cs})
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(rows)), random_state=0).fit(
            rows[self.cols].to_numpy(float))
        self.cats = cats
        self.levels = {c: rows[c].astype(str).value_counts().index[:10].tolist() for c in cats}
        return self

    def transform(self, rows):
        Q = pd.DataFrame(self.qt.transform(rows[self.cols].to_numpy(float)), columns=self.cols)
        parts = [Q[cs].mean(1).to_numpy()[:, None] for _, cs in sorted(self.groups.items())]
        if self.src and self.dst:
            parts.append((Q[self.src].mean(1) - Q[self.dst].mean(1)).to_numpy()[:, None])
        for c in self.cats:
            v = rows[c].astype(str).to_numpy()
            parts.append(0.5 * (v[:, None] == np.array(self.levels[c])[None]).astype(float))
        return np.hstack(parts)

    def describe(self):
        d = {g: cs for g, cs in sorted(self.groups.items())}
        if self.src and self.dst:
            d["direction"] = {"source": self.src, "destination": self.dst}
        return d


def behaviour_types(abnormal, nums, cats, seed, k_range=None, k_fixed=None):
    """Cluster anomalies in behaviour space. k chosen by silhouette unless fixed."""
    space = BehaviourSpace().fit(abnormal, nums, cats)
    Z = space.transform(abnormal)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z), min(len(Z), 4000), replace=False)
    best = None
    ks = [k_fixed] if k_fixed else range(k_range[0], k_range[1] + 1)
    for k in ks:
        km = KMeans(k, n_init=5, random_state=seed).fit(Z)
        sil = silhouette_score(Z[sub], km.labels_[sub]) if len(ks) > 1 else 0.0
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best
    return km.labels_, km.cluster_centers_, dict(k=k, silhouette=round(float(sil), 3), space=space.describe())


def pick_novel(labels, centers, share, max_share=0.45, min_size=0.03, max_types_frac=0.34, rotation=0):
    """Hold out the types whose behaviour is farthest from every other type
    until about `share` of the anomalies are hidden. Only types holding at
    least `min_size` of the anomalies qualify (tiny outlier clusters would make
    the pseudo-novel AUPRC noise), and at most a third of the types are hidden.
    `rotation` r starts from the r-th block of the novelty ranking, so different
    splits hide different types (wrapping round when the ranking runs out)."""
    k = len(centers)
    D = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(D, np.inf)
    novelty = D.min(1)
    frac = np.bincount(labels, minlength=k) / len(labels)
    max_types = max(1, int(round(max_types_frac * k)))
    order = [t for t in np.argsort(-novelty) if frac[t] >= min_size]
    start = (rotation * max_types) % max(1, len(order))
    order = order[start:] + order[:start]
    hidden, tot = [], 0.0
    for t in order:
        if tot >= share or len(hidden) >= max_types:
            break
        if tot + frac[t] > max_share and hidden:
            continue
        hidden.append(int(t))
        tot += frac[t]
    return hidden, float(tot)


def make_behaviour_split(df, nums, cats, seed, rotation=0, cfg_p=P17):
    """Train side: normals + seen-type anomalies. Validation side: a share of
    normals, a share of seen-type anomalies and ALL pseudo-novel anomalies."""
    rng = np.random.default_rng(seed)
    an = df[df[LABEL] == 1]
    labels, centers, meta = behaviour_types(an, nums, cats, seed, cfg_p["behaviour_k"])
    hidden, share = pick_novel(labels, centers, cfg_p["hidden_share"], rotation=rotation)
    t = pd.Series(-1, index=df.index)
    t[an.index] = labels
    is_an = (df[LABEL] == 1).to_numpy()
    novel = t.isin(hidden).to_numpy()
    u = rng.random(len(df))
    to_val = novel | (~is_an & (u < cfg_p["val_normal_frac"])) | (is_an & ~novel & (u < cfg_p["val_seen_frac"]))
    val = df[to_val].copy()
    val["_unseen"] = novel[to_val]
    meta.update(rotation=rotation, hidden_types=hidden, hidden_share=round(share, 3),
                type_sizes=np.bincount(labels).tolist(), val_rows=int(to_val.sum()))
    return df[~to_val], val, meta


class BehaviourAnomalyModel(p16.AnomalyModel):
    """Pipeline 16 anomaly model; in "loto" mode the left-out groups are
    behaviour types instead of k-means clusters on raw features."""

    def fit(self, normal, abnormal, feats, cats, seed, mode="standard"):
        if mode != "loto" or len(abnormal) < 20 * p16.LOTO_TYPES:
            return super().fit(normal, abnormal, feats, cats, seed, mode)
        self.enc = CodeEncoder(feats, cats, [normal, abnormal])
        Xn, Xa = self.enc(normal), self.enc(abnormal)
        nums = [f for f in feats if f not in cats]
        types, _, _ = behaviour_types(abnormal, nums, cats, seed, k_fixed=p16.LOTO_TYPES)
        groups = [types != t for t in range(p16.LOTO_TYPES)]
        folds = list(KFold(5, shuffle=True, random_state=seed).split(Xn))
        P, self.models = [], []
        for g in groups:
            A = Xa[g]
            p = np.zeros(len(Xn))
            for tr, te in folds:
                X = np.vstack([Xn[tr], A])
                y = np.r_[np.zeros(len(tr)), np.ones(len(A))]
                p[te] = self._clf(seed).fit(X, y).predict_proba(Xn[te])[:, 1]
            P.append(p)
            self.models.append(self._clf(seed).fit(np.vstack([Xn, A]), np.r_[np.zeros(len(Xn)), np.ones(len(A))]))
        self.p_normal_oof = np.mean(P, axis=0)
        return self


if P17["behaviour_loto"]:
    p16.AnomalyModel = BehaviourAnomalyModel     # prepare() picks it up


def ecod_scores_pyod(Xtr, Xte):
    """ECOD exactly as PyOD scores new data (pyod/models/ecod.py, decision_function):
    ECDFs on train ∪ test; per column O = max(U_skew, max(U_l, U_r)); sum over columns.
    Pipeline 16's fallback used max(U_skew, (U_l + U_r)/2), which ranked our Kaggle
    submissions with Spearman −0.19 vs +0.83 for this rule (12 UNSW files, one-hot)."""
    X = np.vstack([Xtr, Xte]).astype(float)
    U_l, U_r = -np.log(p16._ecdf_cols(X)), -np.log(p16._ecdf_cols(-X))
    with np.errstate(all="ignore"):
        sk = np.nan_to_num(np.sign(p16.skew(X, axis=0)))
    U_skew = U_l * -1 * np.sign(sk - 1) + U_r * np.sign(sk + 1)
    return np.maximum(U_skew, np.maximum(U_l, U_r)).sum(1)[-len(Xte):]


if ECOD_RULE == "pyod":
    p16.ecod_scores = ecod_scores_pyod
    ecod_scores = ecod_scores_pyod


# =====================================================================
# 3. SELECTION (Pipeline 16 trim + sculpt, returning pool indices)
# =====================================================================
def select_indices(ctx, S, n_target, seed):
    """Same trim + sculpt as pipeline16.finalize, but stops before spreading and
    returns pool positions so rows can be swapped afterwards."""
    rng = np.random.default_rng(seed + 23)
    spec = ctx["spec"]
    w = weights_for(ctx["p"], S)
    quotas = make_quotas(ctx["keys_real"], w, n_target)
    comp = ctx["components"]
    n_pool = len(ctx["pool"])
    score = np.mean([rankdata(comp[nm]) / n_pool for nm in S["scorer"].split("+")], axis=0)
    sel_by_t, kept_by_t = {}, {}
    for t, q in quotas.items():
        pos = ctx["pos_by_key"].get(t, np.array([], dtype=int))
        sel_by_t[t] = pos[: min(len(pos), max(MIN_CANDIDATES, int(S["overgen"] * q)))]
        k = int(np.ceil(S["keep_frac"] * len(sel_by_t[t])))
        kept_by_t[t] = sel_by_t[t][np.argsort(score[sel_by_t[t]])[:k]]
    kept_all = np.concatenate([v for v in kept_by_t.values() if len(v)])
    ratio = np.ones(n_pool)
    if S["sculpt"] == "sir":
        ratio[kept_all] = density_ratio(ctx["normal"], w, ctx["pool"].iloc[kept_all], ctx["enc"], seed)
    final, short = {}, {}
    for t, q in quotas.items():
        kt = kept_by_t.get(t, np.array([], dtype=int))
        if len(kt) >= q:
            final[t] = kt[weighted_choice_noreplace(rng, q, ratio[kt])]
            continue
        extra = np.setdiff1d(sel_by_t[t], kt)
        extra = extra[np.argsort(score[extra])][: q - len(kt)]
        final[t] = np.concatenate([kt, extra])
        if len(final[t]) < q:
            short[t] = q - len(final[t])
    fb = postprocess(ctx["fallback"].sample(short, rng), spec) if short else None
    return final, kept_by_t, fb


# =====================================================================
# 4. B — DETECTOR SURROGATES + JOINT SWAP
# =====================================================================
def ecod_surrogate(normal, abnormal, pool, cols, bins):
    """Per-column log P(bin | anomaly) / P(bin | normal), summed over columns."""
    h = np.zeros(len(pool))
    for c in cols:
        both = np.r_[normal[c].to_numpy(float), abnormal[c].to_numpy(float)]
        edges = np.unique(np.quantile(both, np.linspace(0, 1, bins + 1)[1:-1]))
        if len(edges) == 0:
            continue
        bn = np.searchsorted(edges, normal[c].to_numpy(float), side="right")
        ba = np.searchsorted(edges, abnormal[c].to_numpy(float), side="right")
        m = len(edges) + 1
        pn = (np.bincount(bn, minlength=m) + 1) / (len(bn) + m)
        pa = (np.bincount(ba, minlength=m) + 1) / (len(ba) + m)
        h += np.log(pa / pn)[np.searchsorted(edges, pool[c].to_numpy(float), side="right")]
    return h


def iforest_surrogate(normal, abnormal, pool, enc, seed, trees):
    """ExtraTrees with one random feature per split ≈ random axis-aligned cuts."""
    X = np.nan_to_num(np.vstack([enc(normal), enc(abnormal)]), nan=-1)
    y = np.r_[np.zeros(len(normal)), np.ones(len(abnormal))]
    et = ExtraTreesClassifier(n_estimators=trees, max_features=1, min_samples_leaf=20, class_weight="balanced",
                              n_jobs=-1, random_state=seed).fit(X, y)
    out = []
    for i in range(0, len(pool), 200000):
        out.append(et.predict_proba(np.nan_to_num(enc(pool.iloc[i:i + 200000]), nan=-1))[:, 1])
    return np.concatenate(out)


def surrogate_ranks(ctx, abnormal, nums, seed):
    """Ranks (0–1, high = harmful) of both surrogates over the whole pool."""
    pool, normal = ctx["pool"], ctx["normal"]
    cols = [c for c in nums if c in pool and normal[c].nunique() > 1]
    he = ecod_surrogate(normal, abnormal, pool, cols, P17["nb_bins"])
    hf = iforest_surrogate(normal, abnormal, pool, ctx["enc"], seed, P17["et_trees"])
    n = len(pool)
    return rankdata(he) / n, rankdata(hf) / n


def joint_swap(final, kept_by_t, r_ecod, r_if, alpha, rho):
    """Within each tuple, swap the ρ most harmful selected rows for the least
    harmful unused trimmed candidates (only when the swap lowers harm)."""
    if rho <= 0:
        return final, 0
    h = alpha * r_ecod + (1 - alpha) * r_if
    out, n_sw = {}, 0
    for t, sel in final.items():
        spare = np.setdiff1d(kept_by_t.get(t, np.array([], dtype=int)), sel)
        m = min(int(round(rho * len(sel))), len(spare))
        if m == 0:
            out[t] = sel
            continue
        worst = sel[np.argsort(-h[sel])[:m]]
        best = spare[np.argsort(h[spare])[:m]]
        better = h[best] < h[worst]
        out[t] = np.concatenate([np.setdiff1d(sel, worst[better]), best[better]])
        n_sw += int(better.sum())
    return out, n_sw


# =====================================================================
# 5. C — PER-COLUMN κ PLAN
# =====================================================================
def apply_kappa_plan(syn, spec, plan):
    """Per-column spreading v = lo + u^κ (hi − lo), with TIE-SAFE ranks.

    Pipeline 16 used average ranks, so a block of tied values sits at its mid
    rank: 91% zeros in a column get u ≈ 0.45. At κ = 50 that maps back to the
    minimum (0.45^50 ≈ 0), but at κ = 4 the whole block moves to 4% of the range
    (NSL duration 0 → 1,727). Here u = (min rank − 1)/(n − 1): the lowest block
    stays exactly at the real minimum and every block keeps its order."""
    out = syn.copy()
    n = len(out)
    for c, k in plan.items():
        if not k or k == 0:
            continue
        lo, hi = spec["ranges"][c]
        if k < 0:   # two-point: lower half of rows → real min, upper half → real max
            r = np.argsort(np.argsort(out[c].to_numpy(float), kind="stable"), kind="stable")
            v = np.where(r < n / 2, lo, hi)
            out[c] = v
            continue
        u = (rankdata(out[c].to_numpy(float), method="min") - 1) / max(n - 1, 1)
        v = lo + (u ** float(k)) * (hi - lo)
        out[c] = np.round(v) if c in spec["integer"] else v
    return p16.restore_derived(out, spec)[spec["feats"]]


def kappa_candidates(ctx, S):
    """Weak (|AUC−0.5| < kappa_delta_max), many-valued, unbounded columns."""
    S2 = dict(S, spread_delta=P17["kappa_delta_max"], spread_skew=-np.inf)   # any weak, many-valued, unbounded column
    cols = spread_columns(ctx["col_stats"], S2, ctx["spec"])
    return sorted(cols, key=lambda c: abs(ctx["col_stats"][c][0] - 0.5))


def p16_plan(ctx, S):
    if not S.get("spread"):
        return {}
    return {c: float(S["spread_kappa"]) for c in spread_columns(ctx["col_stats"], S, ctx["spec"])}


def assemble(ctx, final, fb, plan):
    syn = ctx["pool"].iloc[np.concatenate(list(final.values()))]
    if fb is not None:
        syn = pd.concat([syn, fb], ignore_index=True)
    return apply_kappa_plan(syn.reset_index(drop=True), ctx["spec"], plan)


# =====================================================================
# 6. LEARNING ON BEHAVIOUR SPLITS
# =====================================================================
class FixedEncoder:
    """One-hot levels fixed once per split so trial encodings are comparable and fast."""

    def __init__(self, syn, nums, cats):
        self.nums, self.cats = nums, cats
        self.levels = {c: np.array(sorted(syn[c].astype(str).unique())) for c in cats}

    def __call__(self, df):
        parts = [df[self.nums].to_numpy(float)]
        for c in self.cats:
            parts.append((df[c].astype(str).to_numpy()[:, None] == self.levels[c][None]).astype(float))
        return np.hstack(parts)


class Split:
    def __init__(self, df, feats, nums, cats, name, S, gen, seed, rotation=0):
        t0 = time.time()
        self.train, self.val, self.meta = make_behaviour_split(df, nums, cats, seed, rotation)
        n = self.train[self.train[LABEL] == 0][feats].reset_index(drop=True)
        a = self.train[self.train[LABEL] == 1][feats].reset_index(drop=True)
        self.ctx = prepare(gen, n, a, feats, nums, cats, name, S, seed, P17["learn_n"])
        self.final, self.kept, self.fb = select_indices(self.ctx, S, P17["learn_n"], seed)
        self.r_ecod, self.r_if = surrogate_ranks(self.ctx, a, nums, seed)
        self.enc = FixedEncoder(assemble(self.ctx, self.final, self.fb, {}), nums, cats)
        self.Xval = self.enc(self.val)
        y = self.val[LABEL].to_numpy()
        uns = self.val["_unseen"].to_numpy()
        self.y = y
        self.m_seen, self.m_unseen = ~uns, (y == 0) | uns
        self.seed = seed
        print("      split seed {} (rotation {}): {} behaviour types (silhouette {}), pseudo-novel {} = {:.0%} of "
              "anomalies, {:.1f} min".format(seed, rotation, self.meta["k"], self.meta["silhouette"], self.meta["hidden_types"],
                                  self.meta["hidden_share"], (time.time() - t0) / 60), flush=True)

    def score(self, plan, alpha, rho):
        final, _ = joint_swap(self.final, self.kept, self.r_ecod, self.r_if, alpha, rho)
        syn = assemble(self.ctx, final, self.fb, plan)
        A = self.enc(syn)
        e = ecod_scores(A, self.Xval)
        f = iforest_scores(A, self.Xval, *P17["eval_iforest"])
        out = {}
        for tag, m in [("seen", self.m_seen), ("unseen", self.m_unseen)]:
            ae, af = average_precision_score(self.y[m], e[m]), average_precision_score(self.y[m], f[m])
            out[tag], out[tag + "_ecod"], out[tag + "_iforest"] = (ae + af) / 2, ae, af
        return out


def evaluate_on(splits, plan, alpha, rho):
    rs = [sp.score(plan, alpha, rho) for sp in splits]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def objective(m, base):
    gap = (m["seen"] - base["seen"]) - (m["unseen"] - base["unseen"])
    return 0.5 * (m["seen"] + m["unseen"]) - P17["lam"] * max(0.0, gap)


def admissible(m, base):
    return m["unseen"] >= base["unseen"] - P17["unseen_tol"]


def learn(df, feats, nums, cats, name, S, gen, seed, verbose=True):
    """Learn the κ plan and (α, ρ) on behaviour splits of the released data."""
    t0 = time.time()
    splits = [Split(df, feats, nums, cats, name, S, gen, seed + 313 * i, rotation=i) for i in range(P17["n_learn_splits"])]
    plan0 = p16_plan(splits[0].ctx, S)
    base = evaluate_on(splits, plan0, 0.5, 0.0)
    log = [dict(step="P16 baseline", plan=dict(plan0), alpha=None, rho=0.0, J=objective(base, base), **base)]
    if verbose:
        print("      P16 plan {}: seen {:.4f}  pseudo-novel {:.4f}".format(sorted(plan0), base["seen"], base["unseen"]))
    plan, best_m, best_J = dict(plan0), base, objective(base, base)

    # ---- C: per-column κ (coordinate ascent from the Pipeline 16 plan)
    if P17["learn_kappa"]:
        cands = [c for c in kappa_candidates(splits[0].ctx, S) if all(c in kappa_candidates(sp.ctx, S) for sp in splits[1:])]
        cands = sorted(set(cands) | set(plan0), key=lambda c: abs(splits[0].ctx["col_stats"][c][0] - 0.5))
        if verbose:
            print("      κ candidates ({}): {}".format(len(cands), cands), flush=True)
        for _ in range(P17["kappa_passes"]):
            for c in cands:
                for k in P17["kappa_options"]:
                    if plan.get(c, 0.0) == k:
                        continue
                    trial = dict(plan)
                    if k == 0:
                        trial.pop(c, None)
                    else:
                        trial[c] = k
                    m = evaluate_on(splits, trial, 0.5, 0.0)
                    J = objective(m, base)
                    if J > best_J + P17["eps"] and admissible(m, base):
                        plan, best_m, best_J = trial, m, J
                        log.append(dict(step="κ {}={}".format(c, k), plan=dict(plan), alpha=None, rho=0.0, J=J, **m))
                        if verbose:
                            print("      accept κ[{}]={:g}: J {:.4f}  seen {:.4f}  pseudo-novel {:.4f}".format(
                                c, k, J, m["seen"], m["unseen"]), flush=True)

    # ---- B: joint selection (α, ρ) with the learned plan
    alpha, rho = 0.5, 0.0
    if P17["joint_select"]:
        for a_ in P17["alphas"]:
            for r_ in P17["rhos"]:
                if r_ == 0:
                    continue
                m = evaluate_on(splits, plan, a_, r_)
                J = objective(m, base)
                if J > best_J + P17["eps"] and admissible(m, base):
                    alpha, rho, best_m, best_J = a_, r_, m, J
                    log.append(dict(step="joint α={} ρ={}".format(a_, r_), plan=dict(plan), alpha=a_, rho=r_, J=J, **m))
                    if verbose:
                        print("      accept α={} ρ={}: J {:.4f}  seen {:.4f}  pseudo-novel {:.4f}".format(
                            a_, r_, J, m["seen"], m["unseen"]), flush=True)
    if verbose:
        print("      learned in {:.1f} min: plan={} α={} ρ={}  ΔJ={:+.4f}".format(
            (time.time() - t0) / 60, plan, alpha, rho, best_J - objective(base, base)), flush=True)
    return dict(plan=plan, alpha=alpha, rho=rho, base=base, best=best_m, log=log,
                behaviour=[sp.meta for sp in splits])


# =====================================================================
# 7. MODES
# =====================================================================
def run_final(cfg, L=None):
    name, cats = cfg["name"], cfg["cats"]
    gen = GENERATOR[name]
    S = settings_for(name)
    print("\n" + "=" * 78 + "\nPIPELINE 17 [final] — {}  generator={}".format(name, gen) + "\n" + "=" * 78)
    t0 = time.time()
    df, feats, nums = load_data(cfg)
    if L is None:
        print("  learning κ plan and joint selection on behaviour splits …", flush=True)
        L = learn(df, feats, nums, cats, name, S, gen, cfg["seed"] + 7)
    normal_all = df[df[LABEL] == 0][feats].reset_index(drop=True)
    abnormal_all = df[df[LABEL] == 1][feats].reset_index(drop=True)
    print("  fitting on all released rows …", flush=True)
    ctx = prepare(gen, normal_all, abnormal_all, feats, nums, cats, name, S, cfg["seed"], cfg["target_n"], verbose=True)
    final, kept, fb = select_indices(ctx, S, cfg["target_n"], cfg["seed"])
    n_sw = 0
    if L["rho"] > 0:
        r_e, r_f = surrogate_ranks(ctx, abnormal_all, nums, cfg["seed"])
        final, n_sw = joint_swap(final, kept, r_e, r_f, L["alpha"], L["rho"])
    syn = assemble(ctx, final, fb, L["plan"])
    syn = syn.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    copies = validate(syn, normal_all, feats, cats, cfg["target_n"])
    path = OUTDIR / "submission_{}_p17.csv".format(name)
    sha = save_submission(syn, path)
    meta = dict(dataset=name, generator=gen, file=path.name, rows=len(syn), exact_copy_rate=copies, sha256=sha,
                minutes=round((time.time() - t0) / 60, 1), settings=S, p17=P17,
                kappa_plan=L["plan"], alpha=L["alpha"], rho=L["rho"], rows_swapped=n_sw,
                fallback_rows=0 if fb is None else len(fb),
                derived=[d["target"] for d in ctx["spec"]["derived"]],
                learning=dict(base=L["base"], best=L["best"], log=L["log"], behaviour=L["behaviour"]))
    (OUTDIR / "{}_p17_metadata.json".format(name)).write_text(json.dumps(meta, indent=2, default=str))
    print("  saved → {}  rows={:,}  κ plan={}  α={} ρ={} ({:,} rows swapped)  copies={:.2%}".format(
        path.name, len(syn), L["plan"], L["alpha"], L["rho"], n_sw, copies))
    return meta


def run_evaluate(cfg):
    """Learn on N_LEARN_SPLITS splits, then score Pipeline 16 vs 17 on a
    separate behaviour split the learner never saw."""
    name, cats = cfg["name"], cfg["cats"]
    gen = GENERATOR[name]
    S = settings_for(name)
    print("\n" + "=" * 78 + "\nPIPELINE 17 [evaluate] — {}".format(name) + "\n" + "=" * 78)
    df, feats, nums = load_data(cfg)
    L = learn(df, feats, nums, cats, name, S, gen, cfg["seed"] + 7)
    print("  independent evaluation split …", flush=True)
    ev = Split(df, feats, nums, cats, name, S, gen, cfg["seed"] + 9001, rotation=P17["n_learn_splits"])
    rows = []
    variants = [("Pipeline 16", p16_plan(ev.ctx, S), 0.5, 0.0),
                ("+ C learned κ", L["plan"], 0.5, 0.0),
                ("+ B joint selection (P17)", L["plan"], L["alpha"], L["rho"])]
    for tag, plan, a_, r_ in variants:
        m = ev.score(plan, a_, r_)
        rows.append(dict(variant=tag, seen=m["seen"], pseudo_novel=m["unseen"], overall=(m["seen"] + m["unseen"]) / 2,
                         ecod_novel=m["unseen_ecod"], iforest_novel=m["unseen_iforest"]))
    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    tab.to_csv(OUTDIR / "{}_p17_evaluation.csv".format(name), index=False)
    (OUTDIR / "{}_p17_evaluation.json".format(name)).write_text(json.dumps(
        dict(learned=dict(plan=L["plan"], alpha=L["alpha"], rho=L["rho"]), eval_split=ev.meta,
             learning_log=L["log"]), indent=2, default=str))
    del ev   # free the evaluation split before the final fit
    return L


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("P17_DATASETS")          # e.g. P17_DATASETS=UNSW-NB15
    for cfg in DATASETS:
        if only and cfg["name"] not in only.split(","):
            continue
        if MODE == "final":
            run_final(cfg)
        elif MODE == "evaluate":
            run_evaluate(cfg)
        elif MODE == "final+evaluate":     # learn once, evaluate, then write the submission
            run_final(cfg, L=run_evaluate(cfg))
        else:
            raise ValueError("MODE must be 'final', 'evaluate' or 'final+evaluate'")


if __name__ == "__main__":
    main()
