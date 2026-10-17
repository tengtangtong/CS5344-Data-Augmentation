# =====================================================================
# PIPELINE 15 — Over-generate → Trim → Sculpt
# Normal-only synthetic data for fixed ECOD + IForest detectors
#
#   released train.csv (normal + seen anomalies)
#     │
#     ├─ 1. Curation          drop normals whose k-NN are mostly anomalies (ENN)
#     ├─ 2. Derived flow      audit row-local identities on real normals;
#     │                       verified derived + constant columns are REMOVED
#     │                       from the generator and RESTORED after generation
#     ├─ 3. Anomaly model     HistGB: real normal vs is_anomaly rows
#     │                       → out-of-fold P(anom) on real normals (target weights)
#     │                       → P(anom) on every synthetic candidate (trimming)
#     ├─ 4. Generator         one of four, all conditioned on the categorical tuple
#     │      baseline         tuple donor-seed interpolation (no trimming)
#     │      vae              mixed-type conditional VAE (NumPy)
#     │      ctgan            CTGAN conditioned on the tuple (ctgan library)
#     │      gmm              per-tuple Gaussian mixture (KDE for rare tuples)
#     ├─ 5. Over-generate     OVERGEN × quota candidates per tuple
#     ├─ 6. Trim              rank candidates by the trimming score and keep
#     │                       KEEP_FRAC (default 10% → 90% trimmed), per tuple
#     ├─ 7. Sculpt            per-tuple quotas from the anomaly-aware target
#     │                       w = (1 - P(anom))^γ, then sampling-importance-
#     │                       resampling with a target-vs-pool density ratio
#     ├─ 8. Range-spread      weak, heavy-tailed columns (|AUC−0.5| < δ on the
#     │                       labelled data, skew > s) are mapped rank-preservingly
#     │                       onto the full real-normal range: v = lo + u^κ (hi − lo)
#     │                       → big-but-normal flows stop looking isolated to
#     │                       IForest; ECOD's tail ranking shifts to the low end
#     └─ 9. Validate + save   exact schema, no NaN, no invented categories
#
# MODE = "generate"  → 4 submissions per dataset (baseline, vae, ctgan, gmm)
#                      + local proxy scores for each
# MODE = "ablate"    → reads KAGGLE_SCORES, picks the best generator per
#                      dataset, runs a one-factor-at-a-time ablation on the
#                      local proxy, then exports the top configurations as
#                      new submission CSVs
#
# The anomaly label is used only by the anomaly model (trimming + target
# weights) and by the local proxy. No anomaly row is ever generated from,
# copied or interpolated.
#
# Requires: numpy pandas scipy scikit-learn (>=1.0; tested 1.3-style API)
# Optional: ctgan (+ torch) for the CTGAN generator; pyod for exact ECOD
# =====================================================================

import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
LABEL = "is_anomaly"
OUTDIR = Path("pipeline15_output")

MODE = "final"             # "final"    → ONE submission per dataset with the
                           #              Kaggle-validated configuration (below)
                           # "generate" → 4 candidates per dataset (baseline/vae/ctgan/gmm)
                           # "ablate"   → Kaggle-driven one-factor-at-a-time ablation
                           # "trim_ablation" → WHERE to apply the anomaly classifier:
                           #              pre-trim (real input normals) vs post-trim
                           #              (over-generated candidates) vs both vs none

# Generator used by MODE="final" (best public score per dataset)
FINAL_GENERATOR = {"NSL-KDD": "vae", "UNSW-NB15": "vae"}
FAST = False               # True = quick smoke test (tiny epochs / sizes)

# !! target_n MUST equal the submission size required by the Kaggle page !!
DATASETS = [
    dict(name="NSL-KDD", filename="train_NSL-KDD.csv",
         cats=["protocol_type", "service", "flag"], target_n=40000, seed=2026),
    dict(name="UNSW-NB15", filename="train_UNSW-NB15.csv",
         cats=["proto", "service", "state"], target_n=70000, seed=2027),
]

GENERATORS = ["baseline", "vae", "ctgan", "gmm"]

# ---- Fill in after submitting the MODE="generate" CSVs to Kaggle -----------
KAGGLE_SCORES = {
    # public scores of the MODE="generate" files (2026-10-07); CTGAN not yet run
    "NSL-KDD":   {"baseline": 0.92954, "vae": 0.92960, "ctgan": None, "gmm": 0.92882},
    "UNSW-NB15": {"baseline": 0.43329, "vae": 0.52515, "ctgan": None, "gmm": 0.51694},
    # later (vae, scorer=clf, κ=2): spread δ=0.20 → 0.50276, δ=0.15 → 0.54466,
    #                               δ=0.10 → 0.55045  ← final UNSW configuration
    # NSL-KDD vae + spread (10 cols incl. error-rate fractions) → 0.92192 → spread OFF
}

# ---- Default settings (chosen on the local proxy; see ablation) -----------
DEFAULTS = dict(
    curate=True,              # ENN curation of real normals
    derived=True,             # derived-feature flow on/off
    weight_mode="power",      # power: (1-P)^gamma | odds: ((1-P)/P)^odds_k
    gamma=8,                  # target weight (1 - P(anom))^gamma
    odds_k=0.5,               # odds-tilt exponent (weight_mode="odds")
    anom_model="standard",    # standard | loto (leave-one-type-out ensemble)
    prefilter_frac=None,      # PRE-TRIM: drop this fraction of REAL normals with the
                              #   highest OOF P(anom) before fitting the generator
    prefilter_p=None,         # drop real normals with OOF P(anom) >= this
                              #   before fitting the generator (None = off)
    overgen=20,               # candidates per final row
    keep_frac=0.10,           # fraction kept after trimming (0.10 → trim 90%)
    scorer="clf+real",        # trimming score (mean rank): '+'-joined subset of
                              #   clf     P(anomaly) from the is_anomaly classifier
                              #   real    P(synthetic) from a real-vs-synthetic discriminator
                              #   ecod / iforest  typicality vs real normals
    stratified=True,          # trim within each categorical tuple
    sculpt="sir",             # none | quota | sir
    trim_baseline=False,      # baseline = plain donor-seed (no trimming)
    spread=False,             # range-spreading of weak heavy-tailed columns
    spread_delta=0.15,        #   column is "weak" if |AUC(normal vs anomaly) − 0.5| < delta
    spread_skew=3.0,          #   and "heavy-tailed" if skewness on normals > this
    spread_kappa=2.0,         #   v = lo + u^kappa (hi − lo), u = within-column rank
    spread_skip_bounded=True, #   never spread [0,1] fraction columns (error rates etc.)
    k_donor=5,                # baseline neighbours
    vae_latent=16, vae_beta=0.5, vae_sampling="posterior",   # posterior | prior
    gmm_max_components=8, gmm_cov="full",
    ctgan_epochs=100, ctgan_conditional=True,
)
PER_DATASET = {                    # per-dataset overrides (chosen on the local proxy)
    # spread over 10 cols incl. 4 error-rate fractions: local +0.014 but Kaggle 0.92960 → 0.92192 → OFF
    "NSL-KDD": dict(),
    # Kaggle public: 0.52515 → 0.54466 with range-spreading (κ=2) + classifier-only trim score
    # Kaggle: spread δ=0.20 (20 cols) 0.50276 < δ=0.15 (15) 0.54466 < δ=0.10 (7 cols) 0.55045
    "UNSW-NB15": dict(anom_model="loto", scorer="clf", spread=True, spread_kappa=2.0, spread_delta=0.10),
}

VAE_HIDDEN, VAE_LR, VAE_BATCH = 128, 1e-3, 256
VAE_EPOCHS = 8 if FAST else 60
CTGAN_BATCH, CTGAN_PAC, CTGAN_VERBOSE = 500, 10, True
ENN_K, ENN_MAX_ANOM_FRAC = 10, 0.5
DISC_MAX_LEVELS = 10              # integer columns with <= this many levels
MIN_CANDIDATES = 20               # min over-generated rows per tuple
LOTO_TYPES = 5                    # pseudo attack types for anom_model="loto"
PREFILTER_MIN_KEEP = 0.5          # pre-filter never drops more than half the normals
DERIVED_PASS = 0.95               # audit threshold (as in Pipeline 13)

# Local proxy
PROXY_N = 5000 if FAST else 20000       # synthetic rows per proxy run
N_SPLITS_GENERATE = 1                   # proxy splits in MODE="generate"
N_SPLITS_ABLATE = 1 if FAST else 3      # proxy splits in MODE="ablate"
PSEUDO_TYPES, HIDDEN_TYPES = 8, 3
PROXY_INCLUDE_CTGAN = True              # CTGAN proxy retrains CTGAN (slow)

# Ablation
ABLATION_MIN_GAIN = 0.001
EXPORT_TOP_K = 3
ABLATE_GENERATOR_HPARAMS = True
FACTORS = {                       # post-hoc factors (reuse one fitted generator)
    "keep_frac": [1.0, 0.5, 0.25, 0.05],
    "overgen": [10],
    "scorer": ["clf", "real", "clf+ecod+iforest", "ecod+iforest"],
    "stratified": [False],
    "sculpt": ["none", "quota"],
    "gamma": [0, 32, 128, 512],
    "spread": [False, True],
    "spread_kappa": [1.0, 4.0],
    "spread_delta": [0.10, 0.20],
}
RETRAIN_FACTORS = {"derived": [False], "curate": [False],
                   "prefilter_p": [None, 0.01], "anom_model": ["standard", "loto"],
                   "weight_mode": ["power", "odds"], "prefilter_frac": [0.10, 0.25]}
GEN_FACTORS = {
    "vae": {"vae_latent": [8, 32], "vae_beta": [0.25, 1.0], "vae_sampling": ["prior"]},
    "gmm": {"gmm_max_components": [4, 16], "gmm_cov": ["diag"]},
    "ctgan": {"ctgan_epochs": [300], "ctgan_conditional": [False]},
    "baseline": {"trim_baseline": [True], "k_donor": [3, 10]},
}
GEN_FIT_KEYS = {
    "vae": ["vae_latent", "vae_beta"], "gmm": ["gmm_max_components", "gmm_cov"],
    "ctgan": ["ctgan_epochs"], "baseline": ["k_donor"],
}
GEN_SAMPLE_KEYS = {"vae": ["vae_sampling"], "ctgan": ["ctgan_conditional"], "gmm": [], "baseline": []}

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SEP = "␟"   # tuple-key separator (never appears in the data)


# =====================================================================
# 2. DATA + SMALL HELPERS
# =====================================================================
def load_data(cfg):
    for p in [Path(cfg["filename"]), BASE_DIR / cfg["filename"],
              BASE_DIR / "data" / "raw" / cfg["filename"],              # repo convention
              BASE_DIR.parent / "data" / "raw" / cfg["filename"]]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("{} not found; edit DATASETS.".format(cfg["filename"]))
    if df.isna().any().any():
        raise ValueError("Input contains missing values.")
    for c in cfg["cats"]:
        df[c] = df[c].astype(str)
    feats = [c for c in df.columns if c != LABEL]
    nums = [c for c in feats if c not in cfg["cats"]]
    return df, feats, nums


def tuple_key(df, cats):
    k = df[cats[0]].astype(str)
    for c in cats[1:]:
        k = k + SEP + df[c].astype(str)
    return k.to_numpy()


def split_key(keys, cats):
    parts = pd.Series(keys).str.split(SEP, expand=True)
    return {c: parts[i].to_numpy() for i, c in enumerate(cats)}


def settings_for(name, **over):
    s = dict(DEFAULTS)
    s.update(PER_DATASET.get(name, {}))
    s.update(over)
    return s


def target_weights(p, gamma, mode="power", odds_k=0.5, odds_clip=1e-3):
    """Anomaly-aware target weights over real normals.
    power: w = (1 - p)^gamma          → only pushes DOWN attack-like normals
    odds : w = ((1 - p) / p)^odds_k   → likelihood-ratio tilt: target density
           q ∝ p_normal · (p_normal / p_anom)^k, so −log q (≈ what ECOD/IForest
           score) moves toward the Neyman–Pearson ranking; rare-but-safe
           normals are pulled UP as well. Capped at the 99th percentile."""
    if mode == "odds":
        pc = np.clip(p, odds_clip, 1 - odds_clip)
        w = ((1 - pc) / pc) ** float(odds_k)
        w = np.minimum(w, np.quantile(w, 0.99))
    else:
        w = np.clip(1.0 - p, 1e-12, 1.0) ** float(gamma)
    return w / w.sum()


def weights_for(p, S):
    return target_weights(p, S["gamma"], S.get("weight_mode", "power"), S.get("odds_k", 0.5))


def make_quotas(keys, w, n):
    """Largest-remainder allocation of n rows to tuples ∝ summed weight."""
    s = pd.Series(w).groupby(keys).sum()
    raw = s / s.sum() * n
    q = np.floor(raw).astype(int)
    rem = n - int(q.sum())
    if rem > 0:
        q[(raw - q).sort_values(ascending=False).index[:rem]] += 1
    return {k: int(v) for k, v in q.items() if v > 0}


def weighted_choice_noreplace(rng, n_pick, r):
    """Efraimidis–Spirakis weighted sampling without replacement."""
    keys = np.log(rng.random(len(r))) / np.maximum(r, 1e-12)
    return np.argsort(-keys)[:n_pick]


def _ratio(num, den):
    den = den.astype(float)
    return (num.astype(float) / den.where(den != 0)).fillna(0.0)


# =====================================================================
# 3. DERIVED-FEATURE FLOW
# =====================================================================
CANDIDATE_DERIVED = {
    "UNSW-NB15": {
        "tcprtt": (["synack", "ackdat"], lambda d: d["synack"] + d["ackdat"]),
        "smean": (["sbytes", "spkts"], lambda d: _ratio(d["sbytes"], d["spkts"])),
        "dmean": (["dbytes", "dpkts"], lambda d: _ratio(d["dbytes"], d["dpkts"])),
        "sload": (["sbytes", "dur"], lambda d: _ratio(8.0 * d["sbytes"], d["dur"])),
        "dload": (["dbytes", "dur"], lambda d: _ratio(8.0 * d["dbytes"], d["dur"])),
        "rate": (["spkts", "dpkts", "dur"], lambda d: _ratio(d["spkts"] + d["dpkts"], d["dur"])),
    },
    "NSL-KDD": {
        "diff_srv_rate": (["same_srv_rate"], lambda d: 1.0 - d["same_srv_rate"]),
        "dst_host_diff_srv_rate": (["dst_host_same_srv_rate"], lambda d: 1.0 - d["dst_host_same_srv_rate"]),
    },
}
ROUNDINGS = {"exact": lambda v: v, "round0": np.round, "round2": lambda v: np.round(v, 2)}


def audit_derived(normal, dataset):
    rows, selected = [], []
    for target, (inputs, fn) in CANDIDATE_DERIVED.get(dataset, {}).items():
        if target not in normal or not set(inputs) <= set(normal):
            continue
        if any(i in [s["target"] for s in selected] for i in inputs):
            continue
        actual = normal[target].to_numpy(float)
        expected = fn(normal).to_numpy(float)
        best = (None, -1.0)
        for name, rf in ROUNDINGS.items():
            e = rf(expected)
            d = np.abs(actual - e)
            ok = (d <= 1e-6) | (d <= 1e-2 * np.maximum(np.abs(actual), np.abs(e)))
            if ok.mean() > best[1]:
                best = (name, float(ok.mean()))
        sel = best[1] >= DERIVED_PASS
        rows.append(dict(feature=target, inputs="+".join(inputs), rounding=best[0],
                         pass_fraction=round(best[1], 4), selected=sel))
        if sel:
            selected.append(dict(target=target, rounding=best[0]))
    return selected, pd.DataFrame(rows)


def build_spec(normal, feats, nums, cats, dataset, use_derived):
    derived, report = audit_derived(normal, dataset) if use_derived else ([], pd.DataFrame())
    dnames = [d["target"] for d in derived]
    constants = {c: float(normal[c].iloc[0]) for c in nums if normal[c].nunique() == 1}
    gen_nums = [c for c in nums if c not in dnames and c not in constants]
    integer = set(c for c in nums if np.allclose(normal[c], np.round(normal[c])))
    disc = [c for c in gen_nums if c in integer and normal[c].nunique() <= DISC_MAX_LEVELS]
    return dict(
        dataset=dataset, feats=feats, nums=nums, cats=cats, derived=derived, derived_report=report,
        constants=constants, gen_nums=gen_nums, disc=disc, cont=[c for c in gen_nums if c not in disc],
        integer=integer,
        ranges={c: (float(normal[c].min()), float(normal[c].max())) for c in nums},
        disc_levels={c: np.sort(normal[c].unique().astype(float)) for c in disc},
        cat_levels={c: sorted(normal[c].astype(str).unique()) for c in cats},
    )


def _snap(v, levels):
    i = np.clip(np.searchsorted(levels, v), 1, len(levels) - 1)
    left, right = levels[i - 1], levels[i]
    return np.where(np.abs(v - left) <= np.abs(right - v), left, right) if len(levels) > 1 else np.full_like(v, levels[0])


def postprocess(raw, spec):
    """Clip/round generated columns → restore constants + derived → schema."""
    out = pd.DataFrame(index=np.arange(len(raw)))
    for c in spec["cats"]:
        out[c] = raw[c].astype(str).to_numpy()
    for c in spec["gen_nums"]:
        lo, hi = spec["ranges"][c]
        v = np.clip(np.nan_to_num(pd.to_numeric(raw[c], errors="coerce").to_numpy(float), nan=lo), lo, hi)
        if c in spec["disc_levels"]:
            v = _snap(v, spec["disc_levels"][c])
        elif c in spec["integer"]:
            v = np.round(v)
        out[c] = v
    for c, v in spec["constants"].items():
        out[c] = v
    restore_derived(out, spec)
    return out[spec["feats"]]


def restore_derived(out, spec):
    for d in spec["derived"]:
        c = d["target"]
        fn = CANDIDATE_DERIVED[spec["dataset"]][c][1]
        v = ROUNDINGS[d["rounding"]](fn(out).to_numpy(float))
        lo, hi = spec["ranges"][c]
        v = np.clip(np.nan_to_num(v, nan=lo, posinf=hi, neginf=lo), lo, hi)
        out[c] = np.round(v) if c in spec["integer"] else v
    return out


def column_stats(normal, abnormal, spec):
    """Univariate AUC (normal vs seen anomaly) and skewness on normals for
    every continuous generated column — used to pick columns to range-spread."""
    y = np.r_[np.zeros(len(normal)), np.ones(len(abnormal))]
    st = {}
    for c in spec["cont"]:
        x = np.r_[normal[c].to_numpy(float), abnormal[c].to_numpy(float)]
        auc = roc_auc_score(y, x) if np.ptp(x) > 0 else 0.5
        st[c] = (auc, float(normal[c].skew()), int(normal[c].nunique()))
    return st


def spread_columns(stats, S, spec=None):
    """Weak (|AUC-0.5| < delta), heavy-tailed (skew > s), many-valued columns.
    Bounded fraction columns (real-normal range inside [0, 1], e.g. NSL-KDD's
    *_error_rate) are skipped: their high tail is a semantic attack signature
    (e.g. REJ-heavy probing) that may be rare among SEEN anomalies but present
    in unseen types — spreading them cost 0.008 on the NSL-KDD public board."""
    out = []
    for c, (auc, sk, nu) in stats.items():
        if not (abs(auc - 0.5) < S["spread_delta"] and sk > S["spread_skew"] and nu > DISC_MAX_LEVELS):
            continue
        if spec is not None and S.get("spread_skip_bounded", True):
            lo, hi = spec["ranges"][c]
            if lo >= 0.0 and hi <= 1.0:
                continue
        out.append(c)
    return out


def apply_spread(syn, spec, cols, kappa):
    """Rank-preserving map of each column onto [lo, hi] of the real normals:
    v = lo + u^kappa (hi - lo). Keeps every row's within-column ordering (the
    joint/copula structure) but flattens the marginal across the normal range."""
    out = syn.copy()
    n = len(out)
    for c in cols:
        u = (rankdata(out[c].to_numpy(float), method="average") - 0.5) / n
        lo, hi = spec["ranges"][c]
        v = lo + (u ** float(kappa)) * (hi - lo)
        out[c] = np.round(v) if c in spec["integer"] else v
    return restore_derived(out, spec)


# =====================================================================
# 4. FIXED DETECTORS + LOCAL PROXY
# =====================================================================
def _ecdf_cols(X):
    out = np.empty_like(X, dtype=float)
    n = len(X)
    for j in range(X.shape[1]):
        s = np.sort(X[:, j])
        out[:, j] = np.searchsorted(s, X[:, j], side="right") / n
    return out


def ecod_scores(Xtr, Xte):
    try:
        from pyod.models.ecod import ECOD
        return ECOD().fit(Xtr).decision_function(Xte)
    except ImportError:
        X = np.vstack([Xtr, Xte]).astype(float)
        U_l, U_r = -np.log(_ecdf_cols(X)), -np.log(_ecdf_cols(-X))
        with np.errstate(all="ignore"):
            sk = np.nan_to_num(np.sign(skew(X, axis=0)))
        U_skew = U_l * -1 * np.sign(sk - 1) + U_r * np.sign(sk + 1)
        return np.maximum(U_skew, (U_l + U_r) / 2).sum(1)[-len(Xte):]


def iforest_scores(Xtr, Xte, seeds=10, trees=500):
    s = np.zeros(len(Xte))
    for k in range(seeds):
        s -= IsolationForest(n_estimators=trees, random_state=k, n_jobs=-1).fit(Xtr).score_samples(Xte)
    return s / seeds


def encode(syn, test, nums, cats, mode):
    A, B = [syn[nums].to_numpy(float)], [test[nums].to_numpy(float)]
    for c in cats:
        lv = np.array(sorted(syn[c].astype(str).unique()))
        if mode == "onehot":
            A.append((syn[c].to_numpy()[:, None] == lv[None]).astype(float))
            B.append((test[c].to_numpy()[:, None] == lv[None]).astype(float))
        else:
            mp = {v: i for i, v in enumerate(lv)}
            A.append(syn[c].map(mp).fillna(-1).to_numpy(float)[:, None])
            B.append(test[c].map(mp).fillna(-1).to_numpy(float)[:, None])
    return np.hstack(A), np.hstack(B)


def proxy_score(syn, test, nums, cats):
    y = test[LABEL].to_numpy()
    unseen = test["_unseen"].to_numpy()
    r, uns = {}, []
    for mode in ["onehot", "ordinal"]:
        A, B = encode(syn, test, nums, cats, mode)
        e, f = ecod_scores(A, B), iforest_scores(A, B, 3, 200)
        r["ecod_" + mode] = average_precision_score(y, e)
        r["iforest_" + mode] = average_precision_score(y, f)
        m = (y == 0) | unseen
        uns.append((average_precision_score(y[m], e[m]) + average_precision_score(y[m], f[m])) / 2)
    r["S"] = float(np.mean([r[k] for k in list(r)]))
    r["S_unseen"] = float(np.mean(uns))
    return r


def make_split(df, nums, cats, seed):
    """30% held out; anomalies clustered into pseudo-types, HIDDEN_TYPES of
    which never reach the training side (mimics unseen A_val / A_test)."""
    rng = np.random.default_rng(seed)
    an = df[df[LABEL] == 1]
    Z = pd.get_dummies(an[cats]).to_numpy(float)
    q = QuantileTransformer(n_quantiles=200, random_state=0).fit_transform(an[nums])
    types = KMeans(PSEUDO_TYPES, n_init=5, random_state=seed).fit_predict(np.hstack([q, Z]))
    hidden = rng.choice(PSEUDO_TYPES, HIDDEN_TYPES, replace=False)
    t = pd.Series(-1, index=df.index)
    t[an.index] = types
    te = rng.random(len(df)) < 0.3
    test = df[te].copy()
    test["_unseen"] = t[te].isin(hidden).to_numpy()
    return df[~te & ~t.isin(hidden).to_numpy()], test


# =====================================================================
# 5. CURATION, ANOMALY MODEL, TRIMMING SCORERS
# =====================================================================
def curate(normal, abnormal, nums, cats):
    full = pd.concat([normal, abnormal], ignore_index=True)
    y = np.r_[np.zeros(len(normal)), np.ones(len(abnormal))]
    Q = QuantileTransformer(n_quantiles=1000, random_state=0).fit_transform(full[nums])
    F = np.hstack([Q, pd.get_dummies(full[cats]).to_numpy(float)])
    _, ind = NearestNeighbors(n_neighbors=ENN_K + 1).fit(F).kneighbors(F[: len(normal)])
    keep = y[ind[:, 1:]].mean(1) <= ENN_MAX_ANOM_FRAC
    return normal[keep].reset_index(drop=True), int((~keep).sum())


class CodeEncoder:
    """Numerics as-is + integer-coded categoricals (unseen → NaN) for HistGB.
    Uses a boolean categorical mask, which works on scikit-learn >= 1.0."""

    def __init__(self, feats, cats, frames):
        self.feats, self.cats = feats, cats
        allv = pd.concat([f[cats] for f in frames], ignore_index=True)
        self.maps = {c: {v: i for i, v in enumerate(sorted(allv[c].astype(str).unique()))} for c in cats}
        self.mask = np.array([f in cats for f in feats])

    def __call__(self, df):
        X = df[self.feats].copy()
        for c in self.cats:
            X[c] = X[c].astype(str).map(self.maps[c])
        return X.to_numpy(float)


class AnomalyModel:
    """HistGB on real normal (0) vs is_anomaly (1).

    mode="standard": one classifier on all seen anomalies.
    mode="loto"    : leave-one-type-out ensemble. Seen anomalies are clustered
                     into LOTO_TYPES pseudo-types; model t is trained without
                     type t, and P(anomaly) is the mean over the models. No
                     single attack type can dominate the weights, which keeps
                     them closer to what an unseen type would produce.
    P(anomaly) on real normals is always out-of-fold (5-fold over normals)."""

    def fit(self, normal, abnormal, feats, cats, seed, mode="standard"):
        self.enc = CodeEncoder(feats, cats, [normal, abnormal])
        Xn, Xa = self.enc(normal), self.enc(abnormal)
        groups = [np.ones(len(abnormal), dtype=bool)]
        if mode == "loto" and len(abnormal) >= 20 * LOTO_TYPES:
            nums = [f for f in feats if f not in cats]
            Q = QuantileTransformer(n_quantiles=min(1000, len(abnormal)), random_state=0).fit_transform(
                abnormal[nums].to_numpy(float))
            Z = np.hstack([Q, pd.get_dummies(abnormal[cats]).to_numpy(float)])
            types = KMeans(LOTO_TYPES, n_init=5, random_state=seed).fit_predict(Z)
            groups = [types != t for t in range(LOTO_TYPES)]
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

    def _clf(self, seed):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed,
                                              categorical_features=self.enc.mask)

    def predict(self, df, chunk=200000):
        out = []
        for i in range(0, len(df), chunk):
            X = self.enc(df.iloc[i:i + chunk])
            out.append(np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0))
        return np.concatenate(out)


class Typicality:
    """ECOD (out-of-sample ECDF tails) and IForest fitted on REAL normals."""

    def fit(self, normal, spec, seed):
        self.spec = spec
        X = self._X(normal)
        self.sorted, self.n = np.sort(X, axis=0), len(X)
        with np.errstate(all="ignore"):
            self.sk = np.nan_to_num(np.sign(skew(X, axis=0)))
        self.iforest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(X)
        return self

    def _X(self, df):
        parts = [df[self.spec["nums"]].to_numpy(float)]
        for c in self.spec["cats"]:
            lv = np.array(self.spec["cat_levels"][c])
            parts.append((df[c].astype(str).to_numpy()[:, None] == lv[None]).astype(float))
        return np.hstack(parts)

    def scores(self, df, chunk=200000):
        e_all, f_all = [], []
        n = self.n
        for i in range(0, len(df), chunk):
            X = self._X(df.iloc[i:i + chunk])
            Ul, Ur = np.empty_like(X), np.empty_like(X)
            for j in range(X.shape[1]):
                s = self.sorted[:, j]
                Ul[:, j] = -np.log(np.clip(np.searchsorted(s, X[:, j], "right") / n, 1.0 / (n + 1), 1))
                Ur[:, j] = -np.log(np.clip((n - np.searchsorted(s, X[:, j], "left")) / n, 1.0 / (n + 1), 1))
            Us = Ul * -1 * np.sign(self.sk - 1) + Ur * np.sign(self.sk + 1)
            e_all.append(np.maximum(Us, (Ul + Ur) / 2).sum(1))
            f_all.append(-self.iforest.score_samples(X))
        return np.concatenate(e_all), np.concatenate(f_all)


def realism_scores(real, pool, enc, seed, max_train=100000):
    """Cross-fitted real-vs-synthetic discriminator → P(synthetic) per pool row."""
    rng = np.random.default_rng(seed)
    Xr = enc(real)
    half = rng.random(len(pool)) < 0.5
    out = np.zeros(len(pool))
    for m in [half, ~half]:
        tr = np.where(~m)[0]
        if len(tr) > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        Xp = enc(pool.iloc[tr])
        X = np.vstack([Xr, Xp])
        y = np.r_[np.zeros(len(Xr)), np.ones(len(Xp))]
        sw = np.r_[np.full(len(Xr), len(Xp) / len(Xr)), np.ones(len(Xp))]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed,
                                             categorical_features=enc.mask).fit(X, y, sample_weight=sw)
        te = np.where(m)[0]
        for i in range(0, len(te), 200000):
            out[te[i:i + 200000]] = clf.predict_proba(enc(pool.iloc[te[i:i + 200000]]))[:, 1]
    return out


def density_ratio(real, w, pool, enc, seed):
    """Sculpting ratio r(x) = p_target(x) / p_pool(x), cross-fitted on the pool.
    Target = real normals weighted by the anomaly-aware weights w."""
    rng = np.random.default_rng(seed + 1)
    Xr = enc(real)
    half = rng.random(len(pool)) < 0.5
    r = np.ones(len(pool))
    for m in [half, ~half]:
        tr, te = np.where(~m)[0], np.where(m)[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        Xp = enc(pool.iloc[tr])
        X = np.vstack([Xr, Xp])
        y = np.r_[np.ones(len(Xr)), np.zeros(len(Xp))]
        sw = np.r_[w / w.sum() * len(Xp), np.ones(len(Xp))]
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed,
                                             categorical_features=enc.mask).fit(X, y, sample_weight=sw)
        p = np.clip(clf.predict_proba(enc(pool.iloc[te]))[:, 1], 1e-4, 1 - 1e-4)
        r[te] = p / (1 - p)
    return np.clip(r, 0.02, 50.0)


# =====================================================================
# 6. GENERATORS — common API: fit(normal, spec, w, seed); sample(counts, rng)
#    `counts` maps categorical-tuple key → number of rows to generate.
# =====================================================================
def _frame_from_keys(keys, cats):
    return pd.DataFrame(split_key(keys, cats))


class DonorSeed:
    """Baseline: within each categorical tuple, interpolate a seed row towards
    one of its k nearest normal neighbours (log1p space for non-negative
    continuous columns; discrete columns copied from seed or donor)."""

    def __init__(self, k=5, weighted=False):
        self.k, self.weighted = k, weighted

    def fit(self, normal, spec, w, seed):
        self.spec, self.normal = spec, normal.reset_index(drop=True)
        self.w = np.ones(len(normal)) if (w is None or not self.weighted) else np.asarray(w)
        cols = spec["gen_nums"]
        Q = QuantileTransformer(n_quantiles=min(1000, len(normal)), random_state=0).fit_transform(
            normal[cols].to_numpy(float)) if cols else np.zeros((len(normal), 1))
        keys = tuple_key(normal, spec["cats"])
        self.groups = {}
        for t in np.unique(keys):
            idx = np.where(keys == t)[0]
            kk = min(self.k, len(idx) - 1)
            nb = (NearestNeighbors(n_neighbors=kk + 1).fit(Q[idx]).kneighbors(Q[idx])[1][:, 1:]
                  if kk > 0 else np.zeros((len(idx), 1), dtype=int))
            self.groups[t] = (idx, idx[nb] if kk > 0 else idx[:, None])
        self.nonneg = set(c for c in spec["cont"] if normal[c].min() >= 0)
        return self

    def sample(self, counts, rng):
        seeds, donors = [], []
        for t, m in counts.items():
            if t not in self.groups or m <= 0:
                continue
            idx, nb = self.groups[t]
            pw = self.w[idx] / self.w[idx].sum()
            pos = rng.choice(len(idx), m, replace=True, p=pw)
            seeds.append(idx[pos])
            donors.append(nb[pos, rng.integers(0, nb.shape[1], m)])
        seeds, donors = np.concatenate(seeds), np.concatenate(donors)
        A = self.normal.iloc[seeds].reset_index(drop=True)
        B = self.normal.iloc[donors].reset_index(drop=True)
        lam = rng.random(len(A))
        out = A[self.spec["cats"] + self.spec["gen_nums"]].copy()
        for c in self.spec["cont"]:
            a, b = A[c].to_numpy(float), B[c].to_numpy(float)
            out[c] = (np.expm1(np.log1p(a) + lam * (np.log1p(b) - np.log1p(a)))
                      if c in self.nonneg else a + lam * (b - a))
        for c in self.spec["disc"]:
            out[c] = np.where(rng.random(len(A)) < 0.5, A[c], B[c])
        return out


def _relu(x):
    return np.maximum(x, 0)


class CVAE:
    """Mixed-type conditional VAE (NumPy, manual backprop + Adam).
    condition  c = one-hot of each categorical column (protocol/service/flag …)
    continuous → QuantileTransformer(normal) → Gaussian head, learned σ per column
    discrete   → integer columns with ≤10 levels → softmax head per column
    encoder q(z | x, c), decoder p(x | z, c); sampling is conditioned on the
    requested tuple, so every generated row has a real categorical tuple."""

    def __init__(self, latent=16, beta=0.5, sampling="prior", hidden=VAE_HIDDEN, lr=VAE_LR,
                 epochs=VAE_EPOCHS, batch=VAE_BATCH):
        self.l, self.beta, self.sampling = latent, beta, sampling
        self.h, self.lr, self.epochs, self.batch = hidden, lr, epochs, batch

    # ------------------------------------------------------------- encoding
    def _onehot(self, values, levels):
        return (np.asarray(values)[:, None] == np.asarray(levels)[None]).astype(float)

    def _cond(self, cat_values):
        return np.hstack([self._onehot(cat_values[c], self.spec["cat_levels"][c]) for c in self.spec["cats"]])

    def fit(self, normal, spec, w, seed):
        self.spec, self.rng = spec, np.random.default_rng(seed)
        cont, disc = spec["cont"], spec["disc"]
        self.kc = len(cont)
        parts = []
        if cont:
            self.qt = QuantileTransformer(n_quantiles=min(1000, len(normal)), output_distribution="normal",
                                          random_state=0).fit(normal[cont].to_numpy(float))
            parts.append(self.qt.transform(normal[cont].to_numpy(float)))
        self.dslices, s = [], self.kc
        for c in disc:
            lv = spec["disc_levels"][c]
            parts.append(self._onehot(normal[c].to_numpy(float), lv))
            self.dslices.append((s, s + len(lv)))
            s += len(lv)
        Y = np.hstack(parts) if parts else np.zeros((len(normal), 0))
        C = self._cond({c: normal[c].astype(str).to_numpy() for c in spec["cats"]})
        self.Y, self.C = Y, C
        self.keys = tuple_key(normal, spec["cats"])
        self.w = np.ones(len(normal)) if w is None else np.asarray(w, float)
        self._init(Y.shape[1], C.shape[1])
        for ep in range(self.epochs):
            idx = self.rng.permutation(len(Y))
            for i in range(0, len(Y), self.batch):
                b = idx[i:i + self.batch]
                self._step(Y[b], C[b])
        return self

    def _init(self, dy, dc):
        g = lambda a, b: self.rng.normal(0, np.sqrt(2.0 / a), (a, b))
        h, l = self.h, self.l
        self.P = dict(W1=g(dy + dc, h), b1=np.zeros(h), W2=g(h, h), b2=np.zeros(h),
                      Wm=g(h, l) * .1, bm=np.zeros(l), Wv=g(h, l) * .1, bv=np.zeros(l),
                      U1=g(l + dc, h), c1=np.zeros(h), U2=g(h, h), c2=np.zeros(h),
                      Uo=g(h, dy) * .1, co=np.zeros(dy), ls=np.zeros(self.kc))
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.V = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def _encode(self, Y, C):
        P = self.P
        a1 = np.hstack([Y, C]) @ P["W1"] + P["b1"]; h1 = _relu(a1)
        a2 = h1 @ P["W2"] + P["b2"]; h2 = _relu(a2)
        return a1, h1, a2, h2, h2 @ P["Wm"] + P["bm"], np.clip(h2 @ P["Wv"] + P["bv"], -8, 8)

    def _decode(self, z, C):
        P = self.P
        D = np.hstack([z, C])
        e1 = D @ P["U1"] + P["c1"]; d1 = _relu(e1)
        e2 = d1 @ P["U2"] + P["c2"]; d2 = _relu(e2)
        return D, e1, d1, e2, d2, d2 @ P["Uo"] + P["co"]

    def _step(self, Y, C):
        P, n, k = self.P, len(Y), self.kc
        a1, h1, a2, h2, mu, lv = self._encode(Y, C)
        eps = self.rng.standard_normal(mu.shape); sd = np.exp(.5 * lv)
        z = mu + sd * eps
        D, e1, d1, e2, d2, out = self._decode(z, C)
        ls = np.clip(P["ls"], -4, 2); sig2 = np.exp(2 * ls)
        r = Y[:, :k] - out[:, :k]
        g_out = np.zeros_like(out)
        g_out[:, :k] = -r / sig2
        G = {"ls": (1 - r ** 2 / sig2).sum(0) / n}
        for a, b in self.dslices:
            lg = out[:, a:b] - out[:, a:b].max(1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            g_out[:, a:b] = pr - Y[:, a:b]
        g_out /= n
        G["Uo"] = d2.T @ g_out; G["co"] = g_out.sum(0)
        g = (g_out @ P["Uo"].T) * (e2 > 0); G["U2"] = d1.T @ g; G["c2"] = g.sum(0)
        g = (g @ P["U2"].T) * (e1 > 0); G["U1"] = D.T @ g; G["c1"] = g.sum(0)
        gz = (g @ P["U1"].T)[:, : self.l]
        g_mu = gz + self.beta * mu / n
        g_lv = gz * eps * .5 * sd + self.beta * .5 * (np.exp(lv) - 1) / n
        G["Wm"] = h2.T @ g_mu; G["bm"] = g_mu.sum(0)
        G["Wv"] = h2.T @ g_lv; G["bv"] = g_lv.sum(0)
        g = (g_mu @ P["Wm"].T + g_lv @ P["Wv"].T) * (a2 > 0); G["W2"] = h1.T @ g; G["b2"] = g.sum(0)
        g = (g @ P["W2"].T) * (a1 > 0); G["W1"] = np.hstack([Y, C]).T @ g; G["b1"] = g.sum(0)
        self.t += 1
        for kk in P:
            gk = np.clip(G[kk], -5, 5)
            self.M[kk] = .9 * self.M[kk] + .1 * gk
            self.V[kk] = .999 * self.V[kk] + .001 * gk ** 2
            P[kk] -= self.lr * (self.M[kk] / (1 - .9 ** self.t)) / (np.sqrt(self.V[kk] / (1 - .999 ** self.t)) + 1e-8)

    def sample(self, counts, rng, chunk=100000):
        keys = np.concatenate([np.repeat(t, m) for t, m in counts.items() if m > 0])
        out_frames = []
        for i in range(0, len(keys), chunk):
            kb = keys[i:i + chunk]
            catv = split_key(kb, self.spec["cats"])
            C = self._cond(catv)
            if self.sampling == "posterior":
                src = np.empty(len(kb), dtype=int)
                for t in np.unique(kb):
                    pos = np.where(kb == t)[0]
                    idx = np.where(self.keys == t)[0]
                    src[pos] = rng.choice(idx, len(pos), replace=True, p=self.w[idx] / self.w[idx].sum())
                *_, mu, lv = self._encode(self.Y[src], self.C[src])
                z = mu + np.exp(.5 * lv) * rng.standard_normal(mu.shape)
            else:
                z = rng.standard_normal((len(kb), self.l))
            out = self._decode(z, C)[-1]
            df = pd.DataFrame(catv)
            if self.kc:
                m = out[:, :self.kc] + np.exp(np.clip(self.P["ls"], -4, 2)) * rng.standard_normal((len(kb), self.kc))
                df = pd.concat([df, pd.DataFrame(self.qt.inverse_transform(np.clip(m, -5.2, 5.2)),
                                                 columns=self.spec["cont"])], axis=1)
            for (a, b), c in zip(self.dslices, self.spec["disc"]):
                lg = out[:, a:b] - out[:, a:b].max(1, keepdims=True)
                pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
                j = np.minimum((rng.random((len(pr), 1)) > pr.cumsum(1)).sum(1), b - a - 1)
                df[c] = self.spec["disc_levels"][c][j]
            out_frames.append(df)
        return pd.concat(out_frames, ignore_index=True)


class CondGMM:
    """Per-tuple Gaussian mixture in quantile-normal space (components chosen
    by BIC). Tuples with < min_rows rows use a Gaussian KDE (one component per
    real row), i.e. the mixture's limiting case."""

    def __init__(self, max_components=8, cov="full", min_rows=30, kde_bw=0.3):
        self.maxc, self.cov, self.min_rows, self.bw = max_components, cov, min_rows, kde_bw

    def fit(self, normal, spec, w, seed):
        self.spec, cols = spec, spec["gen_nums"]
        X = normal[cols].to_numpy(float)
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(normal)), output_distribution="normal",
                                      random_state=0).fit(X)
        Z = self.qt.transform(X)
        self.sd = Z.std(0) + 1e-3
        keys = tuple_key(normal, spec["cats"])
        self.models = {}
        for t in np.unique(keys):
            Zt = Z[keys == t]
            if len(Zt) < self.min_rows:
                self.models[t] = ("kde", Zt)
                continue
            best, best_bic = None, np.inf
            for k in [1, 2, 4, 8, 16, 32]:
                if k > self.maxc or k > max(1, len(Zt) // self.min_rows):
                    break
                g = GaussianMixture(k, covariance_type=self.cov, reg_covar=1e-2, max_iter=200,
                                    random_state=seed).fit(Zt)
                b = g.bic(Zt)
                if b < best_bic:
                    best, best_bic = g, b
            self.models[t] = ("gmm", best)
        return self

    def sample(self, counts, rng):
        Zs, keys = [], []
        for t, m in counts.items():
            if m <= 0 or t not in self.models:
                continue
            kind, obj = self.models[t]
            if kind == "gmm":
                obj.random_state = int(rng.integers(0, 2 ** 31 - 1))
                Zt = obj.sample(m)[0][rng.permutation(m)]
            else:
                Zt = obj[rng.integers(0, len(obj), m)] + rng.standard_normal((m, obj.shape[1])) * self.sd * self.bw
            Zs.append(Zt)
            keys.append(np.repeat(t, m))
        keys = np.concatenate(keys)
        df = _frame_from_keys(keys, self.spec["cats"])
        num = pd.DataFrame(self.qt.inverse_transform(np.clip(np.vstack(Zs), -5.2, 5.2)),
                           columns=self.spec["gen_nums"])
        return pd.concat([df, num], axis=1)


def ctgan_available():
    try:
        import ctgan  # noqa: F401
        return True
    except ImportError:
        return False


class CTGANGen:
    """CTGAN (Xu et al., 2019) with the categorical tuple as ONE discrete
    column, so its conditional vector + training-by-sampling operate on the
    tuple; sampling uses condition_column='_tuple' (one call per tuple)."""

    def __init__(self, epochs=100, conditional=True):
        self.epochs, self.conditional = epochs, conditional

    def fit(self, normal, spec, w, seed):
        try:
            from ctgan import CTGAN
        except ImportError:
            from ctgan import CTGANSynthesizer as CTGAN
        self.spec = spec
        data = normal[spec["gen_nums"]].copy()
        for c in spec["disc"]:
            data[c] = data[c].round().astype(int).astype(str)
        data["_tuple"] = tuple_key(normal, spec["cats"])
        self.model = CTGAN(epochs=self.epochs, batch_size=CTGAN_BATCH, pac=CTGAN_PAC, verbose=CTGAN_VERBOSE)
        if hasattr(self.model, "set_random_state"):
            self.model.set_random_state(seed)
        self.model.fit(data, discrete_columns=list(spec["disc"]) + ["_tuple"])
        return self

    def sample(self, counts, rng):
        if self.conditional:
            parts = []
            for t, m in counts.items():
                if m <= 0:
                    continue
                try:
                    parts.append(self.model.sample(int(m), condition_column="_tuple", condition_value=t))
                except Exception:
                    parts.append(self.model.sample(int(m)))
            df = pd.concat(parts, ignore_index=True)
        else:
            df = self.model.sample(int(sum(counts.values())))
        for c in self.spec["disc"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        catv = split_key(df["_tuple"].astype(str).to_numpy(), self.spec["cats"])
        for c in self.spec["cats"]:
            df[c] = catv[c]
        return df[self.spec["cats"] + self.spec["gen_nums"]]


def make_generator(name, S):
    if name == "baseline":
        return DonorSeed(k=S["k_donor"])
    if name == "vae":
        return CVAE(latent=S["vae_latent"], beta=S["vae_beta"], sampling=S["vae_sampling"])
    if name == "gmm":
        return CondGMM(max_components=S["gmm_max_components"], cov=S["gmm_cov"])
    if name == "ctgan":
        return CTGANGen(epochs=S["ctgan_epochs"], conditional=S["ctgan_conditional"])
    raise ValueError(name)


# =====================================================================
# 7. PREPARE (fit everything, over-generate, score)  /  FINALIZE (trim+sculpt)
# =====================================================================
def prepare(gen_name, normal_raw, abnormal, feats, nums, cats, dataset, S, seed, n_target,
            overgen_max=None, gen_cache=None, verbose=False):
    t0 = time.time()
    is_base = gen_name == "baseline"
    if S["curate"] and not is_base:
        normal, n_removed = curate(normal_raw, abnormal, nums, cats)
    else:
        normal, n_removed = normal_raw.reset_index(drop=True), 0
    spec = build_spec(normal, feats, nums, cats, dataset, S["derived"])
    ctx = dict(gen_name=gen_name, normal=normal, spec=spec, curated_removed=n_removed,
               keys_real=tuple_key(normal, cats), seed=seed)
    fallback = DonorSeed(k=5).fit(normal, spec, None, seed)
    ctx["fallback"] = fallback
    if is_base and not S["trim_baseline"]:
        ctx["gen"] = DonorSeed(k=S["k_donor"]).fit(normal, spec, None, seed) if S["k_donor"] != 5 else fallback
        ctx["fit_seconds"] = time.time() - t0
        return ctx

    ctx["col_stats"] = column_stats(normal, abnormal, spec)
    am = AnomalyModel().fit(normal, abnormal, feats, cats, seed, mode=S["anom_model"])
    p = am.p_normal_oof
    ctx["prefiltered_removed"] = 0
    if S["prefilter_p"] is not None:
        thr = max(float(S["prefilter_p"]), float(np.quantile(p, PREFILTER_MIN_KEEP)))
        keep = p < thr if thr > 0 else p <= 0
        if keep.sum() < len(p):
            ctx["prefiltered_removed"] = int((~keep).sum())
            normal, p = normal[keep].reset_index(drop=True), p[keep]
            ctx["normal"], ctx["keys_real"] = normal, tuple_key(normal, cats)
            ctx["fallback"] = DonorSeed(k=5).fit(normal, spec, None, seed)
    if S.get("prefilter_frac"):
        # rank real normals by OOF P(anomaly), ties broken at random, drop the top fraction
        k = int(round(float(S["prefilter_frac"]) * len(p)))
        tie = np.random.default_rng(seed + 3).random(len(p))
        order = np.lexsort((tie, -p))                  # highest P first
        keep = np.ones(len(p), dtype=bool)
        keep[order[:k]] = False
        ctx["prefiltered_removed"] = ctx.get("prefiltered_removed", 0) + int(k)
        ctx["prefilter_p_cut"] = float(p[order[k - 1]]) if k > 0 else None
        normal, p = normal[keep].reset_index(drop=True), p[keep]
        ctx["normal"], ctx["keys_real"] = normal, tuple_key(normal, cats)
        ctx["fallback"] = DonorSeed(k=5).fit(normal, spec, None, seed)
    ctx["p"] = p
    w = weights_for(p, S)

    fit_key = (gen_name, S["curate"], S["derived"], S["anom_model"], S["prefilter_p"],
               S.get("prefilter_frac"), seed) + \
        tuple(S[k] for k in GEN_FIT_KEYS[gen_name])
    if gen_cache is not None and fit_key in gen_cache:
        gen = gen_cache[fit_key]
    else:
        if verbose:
            print("      fitting {} …".format(gen_name), flush=True)
        gen = make_generator(gen_name, S).fit(normal, spec, w, seed)
        if gen_cache is not None:
            gen_cache[fit_key] = gen
    if gen_name == "vae":
        gen.sampling, gen.w = S["vae_sampling"], w * len(w)
    if gen_name == "ctgan":
        gen.conditional = S["ctgan_conditional"]
    ctx["gen"] = gen

    og = overgen_max or S["overgen"]
    quotas = make_quotas(ctx["keys_real"], w, n_target)
    counts = {t: max(MIN_CANDIDATES, int(og * q)) for t, q in quotas.items()}
    rng = np.random.default_rng(seed + 11)
    pool = postprocess(gen.sample(counts, rng), spec)
    ctx["pool"] = pool
    ctx["pool_keys"] = tuple_key(pool, cats)
    ctx["pos_by_key"] = pd.Series(np.arange(len(pool))).groupby(ctx["pool_keys"]).apply(np.array).to_dict()
    if verbose:
        print("      over-generated {:,} candidates; scoring …".format(len(pool)), flush=True)
    typ = Typicality().fit(normal, spec, seed)
    e, f = typ.scores(pool)
    ctx["enc"] = CodeEncoder(feats, cats, [normal, abnormal])
    ctx["components"] = dict(clf=am.predict(pool), ecod=e, iforest=f,
                             real=realism_scores(normal, pool, ctx["enc"], seed))
    ctx["fit_seconds"] = time.time() - t0
    return ctx


def finalize(ctx, S, n_target, seed):
    rng = np.random.default_rng(seed + 23)
    spec, cats = ctx["spec"], ctx["spec"]["cats"]
    info = dict(n=n_target, curated_removed=ctx["curated_removed"],
                prefiltered_removed=ctx.get("prefiltered_removed", 0),
                derived=[d["target"] for d in spec["derived"]], constants=list(spec["constants"]))
    if ctx["gen_name"] == "baseline" and not S["trim_baseline"]:
        quotas = make_quotas(ctx["keys_real"], np.ones(len(ctx["keys_real"])), n_target)
        syn = postprocess(ctx["gen"].sample(quotas, rng), spec)
        info.update(pool=n_target, kept=n_target, fallback_rows=0)
        return syn.sample(frac=1, random_state=seed).reset_index(drop=True), info

    w = weights_for(ctx["p"], S)
    quotas = make_quotas(ctx["keys_real"], w, n_target)
    comp = ctx["components"]
    names = S["scorer"].split("+")
    n_pool = len(ctx["pool"])
    score = np.mean([rankdata(comp[nm]) / n_pool for nm in names], axis=0)

    # ---- over-generation budget + TRIM (lowest score = most normal/realistic)
    sel_by_t, kept_by_t = {}, {}
    for t, q in quotas.items():
        pos = ctx["pos_by_key"].get(t, np.array([], dtype=int))
        sel_by_t[t] = pos[: min(len(pos), max(MIN_CANDIDATES, int(S["overgen"] * q)))]
    if S["stratified"]:
        for t, pos in sel_by_t.items():
            k = int(math.ceil(S["keep_frac"] * len(pos)))
            kept_by_t[t] = pos[np.argsort(score[pos])[:k]]
    else:
        allpos = np.concatenate(list(sel_by_t.values()))
        k = int(math.ceil(S["keep_frac"] * len(allpos)))
        kept = allpos[np.argsort(score[allpos])[:k]]
        kk = ctx["pool_keys"][kept]
        kept_by_t = {t: kept[kk == t] for t in sel_by_t}
    kept_all = np.concatenate([v for v in kept_by_t.values() if len(v)])

    # ---- SCULPT
    if S["sculpt"] == "none":
        final = kept_all[rng.choice(len(kept_all), n_target, replace=len(kept_all) < n_target)]
        info.update(pool=int(sum(len(v) for v in sel_by_t.values())), kept=len(kept_all), fallback_rows=0)
        syn = ctx["pool"].iloc[final].reset_index(drop=True)
        syn = _maybe_spread(syn, ctx, S, info)
        return syn.sample(frac=1, random_state=seed).reset_index(drop=True), info

    ratio = np.ones(n_pool)
    if S["sculpt"] == "sir":
        ratio[kept_all] = density_ratio(ctx["normal"], w, ctx["pool"].iloc[kept_all], ctx["enc"], seed)
    final, short = [], {}
    for t, q in quotas.items():
        kt = kept_by_t.get(t, np.array([], dtype=int))
        if len(kt) >= q:
            final.append(kt[weighted_choice_noreplace(rng, q, ratio[kt])])
            continue
        extra = np.setdiff1d(sel_by_t[t], kt)                  # relaxed trimming
        extra = extra[np.argsort(score[extra])][: q - len(kt)]
        final.append(np.concatenate([kt, extra]))
        if len(kt) + len(extra) < q:
            short[t] = q - len(kt) - len(extra)
    syn = ctx["pool"].iloc[np.concatenate(final)]
    n_fb = int(sum(short.values()))
    if n_fb:
        syn = pd.concat([syn, postprocess(ctx["fallback"].sample(short, rng), spec)], ignore_index=True)
    info.update(pool=int(sum(len(v) for v in sel_by_t.values())), kept=len(kept_all), fallback_rows=n_fb)
    syn = _maybe_spread(syn.reset_index(drop=True), ctx, S, info)
    return syn.sample(frac=1, random_state=seed).reset_index(drop=True), info


def _maybe_spread(syn, ctx, S, info):
    if not S.get("spread") or "col_stats" not in ctx:
        info["spread_cols"] = []
        return syn
    cols = spread_columns(ctx["col_stats"], S, ctx["spec"])
    info["spread_cols"] = cols
    return apply_spread(syn, ctx["spec"], cols, S["spread_kappa"])[ctx["spec"]["feats"]]


# =====================================================================
# 8. VALIDATION + SAVE
# =====================================================================
def validate(syn, normal_all, feats, cats, n):
    assert len(syn) == n, "row count {} != {}".format(len(syn), n)
    assert list(syn.columns) == feats, "column order mismatch"
    assert not syn.isna().any().any(), "NaNs present"
    for c in cats:
        bad = set(syn[c]) - set(normal_all[c])
        assert not bad, "invented categories in {}: {}".format(c, bad)
    real = set(pd.util.hash_pandas_object(normal_all[feats], index=False))
    return float(pd.util.hash_pandas_object(syn[feats], index=False).isin(real).mean())


def save_submission(syn, path):
    out = syn.copy()
    out.insert(0, "id", np.arange(len(out)))
    out.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =====================================================================
# 9. MODE = "generate"
# =====================================================================
def proxy_eval(gen_name, df, feats, nums, cats, cfg, S, n_splits, gen_cache=None):
    rows = []
    for k in range(n_splits):
        sseed = cfg["seed"] + 101 * k
        train, test = make_split(df, nums, cats, sseed)
        n_raw = train[train[LABEL] == 0][feats].reset_index(drop=True)
        a_tr = train[train[LABEL] == 1][feats].reset_index(drop=True)
        n_proxy = min(PROXY_N, int(round(cfg["target_n"] * len(n_raw) / (df[LABEL] == 0).sum())))
        ctx = prepare(gen_name, n_raw, a_tr, feats, nums, cats, cfg["name"], S, sseed, n_proxy,
                      overgen_max=S["overgen"], gen_cache=gen_cache)
        syn, _ = finalize(ctx, S, n_proxy, sseed)
        rows.append(dict(split=k, **proxy_score(syn, test, nums, cats)))
    return pd.DataFrame(rows)


def run_generate(cfg):
    name, cats = cfg["name"], cfg["cats"]
    print("\n" + "=" * 78 + "\nPIPELINE 15 [generate] — " + name + "\n" + "=" * 78)
    df, feats, nums = load_data(cfg)
    normal_all = df[df[LABEL] == 0][feats].reset_index(drop=True)
    abnormal_all = df[df[LABEL] == 1][feats].reset_index(drop=True)
    S = settings_for(name)
    spec0 = build_spec(normal_all, feats, nums, cats, name, True)
    print("Derived-feature audit (real normals):")
    print(spec0["derived_report"].to_string(index=False) if len(spec0["derived_report"]) else "  none")
    print("constant columns restored after generation:", list(spec0["constants"]) or "none")
    print("discrete numeric columns (softmax/levels):", spec0["disc"])
    spec0["derived_report"].to_csv(OUTDIR / "{}_derived_audit.csv".format(name), index=False)

    metas, proxy_rows = [], []
    for g in GENERATORS:
        if g == "ctgan" and not ctgan_available():
            print("\n[{}] SKIPPED — `pip install ctgan` to enable".format(g))
            continue
        t0 = time.time()
        print("\n[{}] fitting on all released rows …".format(g), flush=True)
        ctx = prepare(g, normal_all, abnormal_all, feats, nums, cats, name, S, cfg["seed"],
                      cfg["target_n"], verbose=True)
        syn, info = finalize(ctx, S, cfg["target_n"], cfg["seed"])
        copies = validate(syn, normal_all, feats, cats, cfg["target_n"])
        path = OUTDIR / "submission_{}_{}.csv".format(name, g)
        sha = save_submission(syn, path)
        meta = dict(dataset=name, generator=g, file=path.name, exact_copy_rate=copies, sha256=sha,
                    minutes=round((time.time() - t0) / 60, 1), **info,
                    settings={k: S[k] for k in S})
        print("  saved → {}  rows={:,}  pool={:,}  kept after trim={:,}  fallback={}  copies={:.2%}".format(
            path.name, len(syn), info["pool"], info["kept"], info["fallback_rows"], copies))
        if N_SPLITS_GENERATE and (g != "ctgan" or PROXY_INCLUDE_CTGAN):
            pr = proxy_eval(g, df, feats, nums, cats, cfg, S, N_SPLITS_GENERATE)
            meta["proxy_S"], meta["proxy_S_unseen"] = float(pr.S.mean()), float(pr.S_unseen.mean())
            proxy_rows.append(dict(generator=g, **pr.mean(numeric_only=True).to_dict()))
            print("  local proxy: S={:.4f}  S_unseen={:.4f}".format(meta["proxy_S"], meta["proxy_S_unseen"]))
        metas.append(meta)
    if proxy_rows:
        pd.DataFrame(proxy_rows).drop(columns="split").to_csv(OUTDIR / "{}_generate_proxy.csv".format(name), index=False)
    (OUTDIR / "{}_generate_metadata.json".format(name)).write_text(json.dumps(metas, indent=2, default=str))
    return metas


# =====================================================================
# 10. MODE = "ablate"  (Kaggle-score driven)
# =====================================================================
def _label(S, base):
    d = {k: v for k, v in S.items() if base.get(k) != v}
    return "default" if not d else ", ".join("{}={}".format(k, v) for k, v in sorted(d.items()))


def run_ablate(cfg):
    name, cats = cfg["name"], cfg["cats"]
    scores = {g: s for g, s in KAGGLE_SCORES.get(name, {}).items() if s is not None}
    if not scores:
        raise ValueError("Fill KAGGLE_SCORES['{}'] with the public scores first.".format(name))
    best_gen = max(scores, key=scores.get)
    print("\n" + "=" * 78 + "\nPIPELINE 15 [ablate] — {}   best Kaggle generator: {} ({:.5f})".format(
        name, best_gen, scores[best_gen]) + "\n" + "=" * 78)
    pf = OUTDIR / "{}_generate_proxy.csv".format(name)
    if pf.exists() and len(scores) >= 3:
        px = pd.read_csv(pf).set_index("generator")["S"]
        common = [g for g in scores if g in px.index]
        if len(common) >= 3:
            rho = spearmanr([scores[g] for g in common], [px[g] for g in common])[0]
            print("Kaggle vs local-proxy rank agreement (Spearman, {} generators): {:.2f}".format(len(common), rho))

    df, feats, nums = load_data(cfg)
    base = settings_for(name)
    if best_gen == "baseline":
        # plain donor-seed has no trimming stage; test the derived flow,
        # neighbourhood size, and *adding* the trim+sculpt stage
        factors = {"derived": RETRAIN_FACTORS.get("derived", [False])}
    else:
        factors = dict(FACTORS)
        factors.update(RETRAIN_FACTORS)
    if ABLATE_GENERATOR_HPARAMS or best_gen == "baseline":
        factors.update(GEN_FACTORS[best_gen])
    variants = [("default", "-", base)]
    for f, levels in factors.items():
        for lv in levels:
            if lv != base.get(f):
                variants.append((f, lv, dict(base, **{f: lv})))

    retrain_keys = set(RETRAIN_FACTORS) | set(GEN_FIT_KEYS[best_gen]) | set(GEN_SAMPLE_KEYS[best_gen]) | {"trim_baseline"}
    rows = []
    gen_cache = {}
    splits = []
    for k in range(N_SPLITS_ABLATE):
        sseed = cfg["seed"] + 101 * k
        train, test = make_split(df, nums, cats, sseed)
        n_raw = train[train[LABEL] == 0][feats].reset_index(drop=True)
        a_tr = train[train[LABEL] == 1][feats].reset_index(drop=True)
        n_proxy = min(PROXY_N, int(round(cfg["target_n"] * len(n_raw) / (df[LABEL] == 0).sum())))
        splits.append((sseed, test, n_raw, a_tr, n_proxy))

    def evaluate_variant(f, lv, S):
        res = []
        for k, (sseed, test, n_raw, a_tr, n_proxy) in enumerate(splits):
            key = tuple((kk, S[kk]) for kk in sorted(retrain_keys) if kk in S)
            ck = (k,) + key
            if ck not in ctx_cache:
                ctx_cache[ck] = prepare(best_gen, n_raw, a_tr, feats, nums, cats, name, S, sseed, n_proxy,
                                        overgen_max=max(20, S["overgen"]), gen_cache=gen_cache)
            syn, _ = finalize(ctx_cache[ck], S, n_proxy, sseed)
            r = proxy_score(syn, test, nums, cats)
            res.append(r)
            rows.append(dict(factor=f, level=str(lv), split=k, label=_label(S, base), **r))
        m = pd.DataFrame(res).mean()
        print("  {:<22s} {:<18s} S={:.4f}  S_unseen={:.4f}".format(f, str(lv), m.S, m.S_unseen), flush=True)
        return m

    ctx_cache = {}
    results = {}
    for f, lv, S in variants:
        results[(f, str(lv))] = (evaluate_variant(f, lv, S), S)
        if f in RETRAIN_FACTORS or f in GEN_FIT_KEYS[best_gen]:
            for ck in [c for c in ctx_cache if any(x == (f, lv) for x in c[1:])]:
                del ctx_cache[ck]          # free memory of one-off retrains

    d0 = results[("default", "-")][0].S
    combo = dict(base)
    for f in factors:
        cand = [(results[(f, str(lv))][0].S, lv) for lv in factors[f] if (f, str(lv)) in results]
        if cand:
            s, lv = max(cand, key=lambda x: x[0])
            if s - d0 > ABLATION_MIN_GAIN:
                combo[f] = lv
    if combo != base:
        results[("combined", "greedy")] = (evaluate_variant("combined", "greedy", combo), combo)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUTDIR / "{}_{}_ablation_runs.csv".format(name, best_gen), index=False)
    summ = pd.DataFrame([dict(factor=f, level=lv, label=_label(S, base), S=m.S, S_unseen=m.S_unseen,
                              delta_vs_default=m.S - d0)
                         for (f, lv), (m, S) in results.items()]).sort_values("S", ascending=False)
    summ.to_csv(OUTDIR / "{}_{}_ablation_summary.csv".format(name, best_gen), index=False)
    print("\n" + summ.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    # ---- export the top configurations as new Kaggle candidates
    ctx_cache.clear(); gen_cache.clear(); splits.clear()      # free proxy pools before full-size fits
    import gc; gc.collect()
    normal_all = df[df[LABEL] == 0][feats].reset_index(drop=True)
    abnormal_all = df[df[LABEL] == 1][feats].reset_index(drop=True)
    seen, exported = [], []
    for _, r in summ.iterrows():
        S = results[(r.factor, r.level)][1]
        if r.label in seen or r.label == "default":
            continue
        seen.append(r.label)
        rank = len(exported) + 1
        ctx = prepare(best_gen, normal_all, abnormal_all, feats, nums, cats, name, S, cfg["seed"],
                      cfg["target_n"], verbose=True)
        syn, info = finalize(ctx, S, cfg["target_n"], cfg["seed"])
        validate(syn, normal_all, feats, cats, cfg["target_n"])
        path = OUTDIR / "submission_{}_{}_ablation{}.csv".format(name, best_gen, rank)
        sha = save_submission(syn, path)
        exported.append(dict(rank=rank, file=path.name, label=r.label, proxy_S=float(r.S),
                             proxy_delta=float(r.delta_vs_default), sha256=sha, settings=S))
        print("  exported #{} → {}   [{}]".format(rank, path.name, r.label))
        if len(exported) >= EXPORT_TOP_K:
            break
    (OUTDIR / "{}_{}_ablation_exports.json".format(name, best_gen)).write_text(
        json.dumps(exported, indent=2, default=str))
    return exported


def run_final(cfg):
    """Fit on all released rows with the Kaggle-validated settings and write
    submission_<dataset>_final.csv plus a metadata JSON."""
    name, cats = cfg["name"], cfg["cats"]
    g = FINAL_GENERATOR[name]
    S = settings_for(name)
    print("\n" + "=" * 78 + "\nPIPELINE 15 [final] — {}  generator={}".format(name, g) + "\n" + "=" * 78)
    print("settings:", {k: S[k] for k in ["curate", "derived", "anom_model", "gamma", "overgen", "keep_frac",
                                          "scorer", "sculpt", "spread", "spread_delta", "spread_kappa"]})
    t0 = time.time()
    df, feats, nums = load_data(cfg)
    normal_all = df[df[LABEL] == 0][feats].reset_index(drop=True)
    abnormal_all = df[df[LABEL] == 1][feats].reset_index(drop=True)
    ctx = prepare(g, normal_all, abnormal_all, feats, nums, cats, name, S, cfg["seed"], cfg["target_n"], verbose=True)
    syn, info = finalize(ctx, S, cfg["target_n"], cfg["seed"])
    copies = validate(syn, normal_all, feats, cats, cfg["target_n"])
    path = OUTDIR / "submission_{}_final.csv".format(name)
    sha = save_submission(syn, path)
    meta = dict(dataset=name, generator=g, file=path.name, rows=len(syn), exact_copy_rate=copies, sha256=sha,
                minutes=round((time.time() - t0) / 60, 1), settings=S, **info)
    (OUTDIR / "{}_final_metadata.json".format(name)).write_text(json.dumps(meta, indent=2, default=str))
    print("  saved → {}  rows={:,}  derived restored={}  spread cols={}  copies={:.2%}".format(
        path.name, len(syn), info.get("derived"), info.get("spread_cols"), copies))
    return meta


# =====================================================================
# 11. MODE = "trim_ablation" — where should the anomaly classifier act?
# =====================================================================
# The classifier is ALWAYS trained on the released labelled data
# (curated real normals vs is_anomaly rows). The variants differ only in
# WHERE its P(anomaly) is used to remove rows:
#   pre-trim  : on the REAL input normals, before the generator is fitted
#   post-trim : on the OVER-GENERATED candidates (×20 → keep 10%)
# Everything else (derived flow, sculpting, range-spreading) is identical.
TRIM_GENERATORS = ["vae", "gmm", "ctgan"]   # ctgan needs `pip install ctgan` (skipped otherwise)
TRIM_CTGAN_EPOCHS = None   # e.g. 50 to shorten CTGAN in this ablation (None = DEFAULTS value);
                           # CTGAN is refitted once per pre-trim level (0 / 10 / 25%) per split
TRIM_VARIANTS = [   # (tag, overrides)
    ("A_post_only",       dict()),                                              # current final
    ("B_none",            dict(overgen=1, keep_frac=1.0)),
    ("C_pre10_only",      dict(prefilter_frac=0.10, overgen=1, keep_frac=1.0)),
    ("D_pre25_only",      dict(prefilter_frac=0.25, overgen=1, keep_frac=1.0)),
    ("E_pre10_post",      dict(prefilter_frac=0.10)),
    ("F_pre25_post",      dict(prefilter_frac=0.25)),
]
TRIM_PROXY_SPLITS = 1          # local proxy splits per variant (0 = skip proxy)
# Fill after submitting the trim-ablation CSVs, e.g. {"vae": {"A_post_only": 0.55045, ...}}
KAGGLE_TRIM_SCORES = {"NSL-KDD": {}, "UNSW-NB15": {}}


def _prepare_key(S):
    return S.get("prefilter_frac") or 0.0


def run_trim_ablation(cfg):
    name, cats = cfg["name"], cfg["cats"]
    print("\n" + "=" * 78 + "\nPIPELINE 15 [trim_ablation] — " + name + "\n" + "=" * 78)
    df, feats, nums = load_data(cfg)
    n_all = df[df[LABEL] == 0][feats].reset_index(drop=True)
    a_all = df[df[LABEL] == 1][feats].reset_index(drop=True)
    base = settings_for(name)
    rows = []
    for g in TRIM_GENERATORS:
        if g == "ctgan" and not ctgan_available():
            print("[ctgan] skipped — pip install ctgan"); continue
        # group variants that share one fitted context (same pre-trim)
        groups = {}
        for tag, ov in TRIM_VARIANTS:
            S = dict(base, **ov)
            if g == "ctgan" and TRIM_CTGAN_EPOCHS:
                S["ctgan_epochs"] = int(TRIM_CTGAN_EPOCHS)
            groups.setdefault(_prepare_key(S), []).append((tag, S))
        # ---- full-data submissions
        gen_cache = {}
        for key, items in groups.items():
            S0 = items[0][1]
            ctx = prepare(g, n_all, a_all, feats, nums, cats, name, S0, cfg["seed"], cfg["target_n"],
                          overgen_max=max(S["overgen"] for _, S in items), gen_cache=gen_cache, verbose=True)
            for tag, S in items:
                syn, info = finalize(ctx, S, cfg["target_n"], cfg["seed"])
                copies = validate(syn, n_all, feats, cats, cfg["target_n"])
                path = OUTDIR / "submission_{}_{}_trim-{}.csv".format(name, g, tag)
                sha = save_submission(syn, path)
                rows.append(dict(dataset=name, generator=g, variant=tag, file=path.name,
                                 pre_trim_removed=info["prefiltered_removed"], pool=info["pool"],
                                 kept_after_post_trim=info["kept"], spread_cols=len(info.get("spread_cols", [])),
                                 copies=copies, sha256=sha,
                                 kaggle=KAGGLE_TRIM_SCORES.get(name, {}).get(g, {}).get(tag)))
                print("  {:<5s} {:<14s} pre-trim removed={:>5,}  candidates={:>9,}  kept={:>8,}  → {}".format(
                    g, tag, info["prefiltered_removed"], info["pool"], info["kept"], path.name), flush=True)
            del ctx
        gen_cache.clear()
        # ---- local proxy (optional)
        for k in range(TRIM_PROXY_SPLITS):
            sseed = cfg["seed"] + 101 * k
            train, test = make_split(df, nums, cats, sseed)
            n_raw = train[train[LABEL] == 0][feats].reset_index(drop=True)
            a_tr = train[train[LABEL] == 1][feats].reset_index(drop=True)
            n_proxy = min(PROXY_N, int(round(cfg["target_n"] * len(n_raw) / (df[LABEL] == 0).sum())))
            cache = {}
            for key, items in groups.items():
                ctx = prepare(g, n_raw, a_tr, feats, nums, cats, name, items[0][1], sseed, n_proxy,
                              overgen_max=max(S["overgen"] for _, S in items), gen_cache=cache)
                for tag, S in items:
                    syn, _ = finalize(ctx, S, n_proxy, sseed)
                    r = proxy_score(syn, test, nums, cats)
                    for row in rows:
                        if row["generator"] == g and row["variant"] == tag:
                            row.setdefault("_S", []).append(r["S"]); row.setdefault("_U", []).append(r["S_unseen"])
                    print("  proxy split {}  {:<5s} {:<14s} S={:.4f}  S_unseen={:.4f}".format(
                        k, g, tag, r["S"], r["S_unseen"]), flush=True)
                del ctx
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
            OUTDIR / "{}_trim_ablation_partial.csv".format(name), index=False)    # survive long CTGAN runs
    for row in rows:
        if "_S" in row:
            row["proxy_S"] = float(np.mean(row.pop("_S"))); row["proxy_S_unseen"] = float(np.mean(row.pop("_U")))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "{}_trim_ablation.csv".format(name), index=False)
    cols = [c for c in ["generator", "variant", "pre_trim_removed", "kept_after_post_trim", "proxy_S",
                        "proxy_S_unseen", "kaggle"] if c in tab]
    print("\n" + tab[cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    return tab


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for cfg in DATASETS:
        if MODE == "final":
            run_final(cfg)
        elif MODE == "trim_ablation":
            run_trim_ablation(cfg)
        elif MODE == "generate":
            run_generate(cfg)
        elif MODE == "ablate":
            run_ablate(cfg)
        else:
            raise ValueError("MODE must be 'final', 'generate', 'ablate' or 'trim_ablation'")


if __name__ == "__main__":
    main()
