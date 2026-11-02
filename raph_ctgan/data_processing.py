"""
CS5344 - Data processing for synthetic NORMAL data generation (Group 13).

Shared by every generator (CTGAN / VAE / GMM). No model code lives here.

What this file does
-------------------
Before training (prepare):
1. Load train.csv, keep normal rows (is_anomaly == 0).
2. Detect column types: categorical / constant / binary / discrete numeric
   (<= DISCRETE_MAX_UNIQUE distinct values) / integer / continuous.
3. Decide which columns the model sees (constant and derived columns are left out).

After sampling (postprocess + finalize):
4. Put constant columns back, snap discrete columns to values seen in training,
   clip and round the rest, rebuild derived columns from their source columns.
5. Drop rows whose rebuilt columns fall outside the real normal range.
6. Validate against the submission rules and add the id column.

Usage (self-test: real normal rows go through drop-and-rebuild, no model needed)
-----
    python data_processing.py --dataset unsw   --train data/UNSW-NB15/train.csv
    python data_processing.py --dataset nslkdd --train data/NSL-KDD/train.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"

LABEL_COLUMN = "is_anomaly"
RANDOM_SEED = 42

# A non-constant numeric column with this many distinct values or fewer is treated
# as a category by the generator (e.g. UNSW sttl has 12). Chosen from the gap in
# the distinct-value counts of both datasets (see EDA).
DISCRETE_MAX_UNIQUE = 15

# Highest number of decimals checked when detecting a column's rounding precision.
MAX_DECIMALS = 6

# Row counts follow the staff sample codebase.
# force_discrete / force_continuous: manual overrides of the cutoff above.
DATASETS = {
    "nslkdd": {
        "train": DATA_ROOT / "NSL-KDD" / "train.csv",
        "out": HERE / "outputs" / "synthetic_nsl-kdd.csv",
        "n": 40000,
        "derived": [],          # EDA found no exact derived columns in NSL-KDD
        "force_discrete": [],
        "force_continuous": [],
        # heavy right tail (skew > 3): trained as log1p(x), inverted after sampling
        "log_columns": ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised",
                        "num_root", "num_file_creations", "count", "srv_count"],
        # zero flags: a 0/1 column "<col> == 0" trained as a category; after sampling the
        # listed columns are set to exactly 0 for flagged rows (see training_frame).
        # Tested on KDD and switched off: Kaggle AUPRC 0.92116 with the flags against
        # 0.92561 without, and the rare flag value REJ doubled (4% to 8%). To re-test:
        #   "zero_flags": [
        #       {"name": "duration_zero", "when": "duration", "zero": ["duration"]},
        #       {"name": "src_bytes_zero", "when": "src_bytes", "zero": ["src_bytes"]},
        #       {"name": "dst_bytes_zero", "when": "dst_bytes", "zero": ["dst_bytes"]},
        #   ],
        "zero_flags": [],
    },
    "unsw": {
        "train": DATA_ROOT / "UNSW-NB15" / "train.csv",
        "out": HERE / "outputs" / "synthetic_unsw-nb15.csv",
        "n": 70000,
        "derived": ["smean", "dmean", "tcprtt", "rate"],
        "force_discrete": [],
        "force_continuous": [],
        # heavy right tail (skew > 5): trained as log1p(x), inverted after sampling.
        # The ct_* counts (skew 3-6, max about 60) and stcpb/dtcpb are left as they are.
        "log_columns": ["spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
                        "response_body_len", "sload", "dload"],
        # sub-second columns: log1p barely changes values below 1, so scale first,
        # i.e. train log1p(1000 * x) and invert as expm1(z) / 1000.
        "log_scaled": {"dur": 1000, "sinpkt": 1000, "dinpkt": 1000,
                       "sjit": 1000, "djit": 1000, "synack": 1000, "ackdat": 1000},
        # no reply (dpkts == 0): all of these are 0 in 100% of the 2,900 real normal rows
        # with dpkts == 0. dur == 0 is a separate flag (only 7% of no-reply rows, but 98%
        # of dur == 0 rows are no-reply).
        "zero_flags": [
            {"name": "no_reply", "when": "dpkts",
             "zero": ["dpkts", "dbytes", "dloss", "dload", "dinpkt", "djit", "dwin", "dttl",
                      "synack", "ackdat", "response_body_len", "trans_depth"]},
            {"name": "dur_zero", "when": "dur", "zero": ["dur"]},
        ],
        # Change C: CTGAN produced upper tails that were too thin (e.g. dbytes p99 at 6% of
        # the real value). For these columns the top (1 - TAIL_START) of the synthetic rows
        # is lifted, rank for rank, to the real tail quantiles (about 10 summary quantiles
        # per column, halving the remaining tail mass each time; no real rows are copied).
        # Lift only: a value is never lowered. sload, dload, response_body_len are left out
        # (their synthetic tails were already as heavy as the real ones).
        "tail_columns": ["spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss", "dur",
                         "sinpkt", "dinpkt", "sjit", "djit"],
        # bytes are capped at packets x the real maximum mean packet size; packets are raised first
        "byte_caps": {"sbytes": ["spkts", "smean"], "dbytes": ["dpkts", "dmean"]},
    },
}

TAIL_START = 0.95   # the tail is the part of the real column above this quantile


# ============================================================
# 1. Load normal rows
# ============================================================

def load_normal(train_csv):
    """Return (feature_columns, normal_rows) from the labelled training CSV."""
    train = pd.read_csv(train_csv)

    if LABEL_COLUMN not in train.columns:
        raise ValueError(f"Training data must contain '{LABEL_COLUMN}'.")

    feature_columns = [c for c in train.columns if c != LABEL_COLUMN]
    normal = train.loc[train[LABEL_COLUMN] == 0, feature_columns].reset_index(drop=True)

    if len(normal) == 0:
        raise ValueError("No normal samples found.")
    if normal.isna().any().any():
        raise ValueError("Normal training data contains missing values.")

    print(f"Train shape: {train.shape}")
    print(f"Normal rows: {len(normal)}")
    print(f"Features: {len(feature_columns)}")
    return feature_columns, normal


# ============================================================
# 2. Identify feature types
# ============================================================

def detect_decimals(values):
    """Smallest number of decimals (0..MAX_DECIMALS) the column is rounded to, else None."""
    values = np.asarray(values, dtype=float)
    for d in range(MAX_DECIMALS + 1):
        if np.abs(values - np.round(values, d)).max() < 1e-9:
            return d
    return None


def detect_column_types(normal, feature_columns, name):
    """Same idea as the staff starter kit, plus constants and the discrete cutoff."""
    cfg = DATASETS[name]

    categorical = [
        c for c in feature_columns
        if pd.api.types.is_object_dtype(normal[c])
        or pd.api.types.is_string_dtype(normal[c])
        or isinstance(normal[c].dtype, pd.CategoricalDtype)
    ]
    numeric = [c for c in feature_columns if c not in categorical]

    constant, binary, discrete, integer, continuous = {}, [], [], [], []

    for col in numeric:
        values = normal[col].to_numpy()
        n_unique = len(np.unique(values))

        if n_unique == 1:
            constant[col] = normal[col].iloc[0]
        elif col in cfg["force_discrete"]:
            discrete.append(col)
        elif col in cfg["force_continuous"]:
            (integer if np.all(np.isclose(values, np.round(values))) else continuous).append(col)
        elif set(np.unique(values).tolist()).issubset({0, 1}):
            binary.append(col)
        elif n_unique <= DISCRETE_MAX_UNIQUE:
            discrete.append(col)
        elif np.all(np.isclose(values, np.round(values))):
            integer.append(col)
        else:
            continuous.append(col)

    decimals = {c: detect_decimals(normal[c]) for c in continuous}
    decimals = {c: d for c, d in decimals.items() if d is not None}

    print(f"Categorical: {len(categorical)}")
    print(f"Constant: {len(constant)} {list(constant)}")
    print(f"Binary: {len(binary)}")
    print(f"Discrete numeric (<= {DISCRETE_MAX_UNIQUE} values): {len(discrete)}")
    print(f"Integer: {len(integer)}")
    print(f"Continuous: {len(continuous)} (rounded to a fixed precision: {len(decimals)})")

    return {
        "categorical": categorical,
        "constant": constant,      # {column: its constant value}
        "binary": binary,
        "discrete": discrete,
        "integer": integer,
        "continuous": continuous,
        "decimals": decimals,      # {column: decimals} for continuous columns
    }


def prepare(name, train_csv, use_zero_flags=True, use_tails=True):
    """Everything a generator needs: data, column roles, and the column lists to train on."""
    feature_columns, normal = load_normal(train_csv)
    types = detect_column_types(normal, feature_columns, name)

    derived = DATASETS[name]["derived"]
    model_columns = [
        c for c in feature_columns
        if c not in derived and c not in types["constant"]
    ]
    category_like = set(types["categorical"] + types["binary"] + types["discrete"])
    discrete_columns = [c for c in model_columns if c in category_like]

    # log1p only for numeric, non-negative model columns
    # log_columns: {column: scale}, trained as log1p(scale * x)
    log_columns = {c: 1 for c in DATASETS[name].get("log_columns", [])
                   if c in model_columns and c not in category_like}
    log_columns.update({c: s for c, s in DATASETS[name].get("log_scaled", {}).items()
                        if c in model_columns and c not in category_like})
    for c in log_columns:
        if normal[c].min() < 0:
            raise ValueError(f"log1p needs non-negative values; '{c}' has negatives.")

    # zero flags (Change B). use_zero_flags=False reproduces the earlier pipeline.
    zero_flags = []
    if use_zero_flags:
        for flag in DATASETS[name].get("zero_flags", []):
            mask = normal[flag["when"]] == 0
            for col in flag["zero"]:
                if col not in model_columns:
                    raise ValueError(f"Flag '{flag['name']}': '{col}' is not a model column.")
                if (normal.loc[mask, col] != 0).any():
                    raise ValueError(f"Flag '{flag['name']}': '{col}' is not always 0 when "
                                     f"{flag['when']} == 0 in the real normal rows.")
            zero_flags.append(flag)
            print(f"Zero flag '{flag['name']}': {mask.mean():.1%} of real rows; "
                  f"zeroes {len(flag['zero'])} column(s)")
    tail_fits = {}
    if use_tails:
        for col in DATASETS[name].get("tail_columns", []):
            if col not in model_columns:
                raise ValueError(f"Tail column '{col}' is not a model column.")
            x = normal[col].astype(float)
            u = float(x.quantile(TAIL_START))
            exceed = (x[x > u] - u).to_numpy()
            if len(exceed) < 100:
                print(f"Tail '{col}': only {len(exceed)} values above the threshold; skipped")
                continue
            f0 = float((x <= u).mean())
            m = 1 - f0
            # knots at tail depths d = 0, 1, 2, ... : remaining tail mass m * 2**-d
            n_knots = int(np.floor(np.log2(m * len(x))))
            depth = np.arange(n_knots + 1, dtype=float)
            level = f0 + m * (1 - 2.0 ** -depth)
            values = np.quantile(x.to_numpy(), level)
            depth = np.append(depth, n_knots + 1.0)       # last knot: the real maximum
            values = np.append(values, float(x.max()))
            tail_fits[col] = {"f0": f0, "depth": depth,
                              "log_values": np.log1p(np.maximum.accumulate(values))}
        if tail_fits:
            print(f"Tail lift on {len(tail_fits)} column(s) above the real "
                  f"{TAIL_START:.0%} quantile")
    flag_columns = [f["name"] for f in zero_flags]
    discrete_columns = discrete_columns + flag_columns

    print(f"Model trains on {len(model_columns) + len(flag_columns)} columns "
          f"({len(discrete_columns)} discrete incl. {len(flag_columns)} flags, "
          f"{len(log_columns)} log-transformed); "
          f"left out: {len(derived)} derived, {len(types['constant'])} constant")

    return {
        "name": name,
        "normal": normal,
        "feature_columns": feature_columns,
        "types": types,
        "derived": derived,
        "model_columns": model_columns,
        "discrete_columns": discrete_columns,
        "log_columns": log_columns,
        "zero_flags": zero_flags,
        "tail_fits": tail_fits,
        "flag_columns": flag_columns,
        "category_like": category_like,
    }


def training_frame(info):
    """Normal rows restricted to model columns, with log1p applied to heavy-tailed ones."""
    normal = info["normal"]
    df = normal[info["model_columns"]].copy()

    # zero flags: add the 0/1 column; for flagged rows, numeric columns get a neutral
    # placeholder (median of the unflagged rows) so the zeros do not distort the fit.
    # Discrete columns keep their real value (0), which is a normal category for them.
    for flag in info["zero_flags"]:
        mask = normal[flag["when"]] == 0
        df[flag["name"]] = mask.astype(int)
        for col in flag["zero"]:
            if col not in info["category_like"]:
                df.loc[mask, col] = normal.loc[~mask, col].median()

    for col, scale in info["log_columns"].items():
        df[col] = np.log1p(df[col].astype(float) * scale)
    return df


# ============================================================
# 3. Derived columns (rebuilt after generation, never sampled)
# ============================================================
# Formulas verified on the normal rows of the UNSW training set:
#   tcprtt = synack + ackdat                            (exact)
#   smean  = floor(sbytes / spkts + 0.5), 0 if spkts==0 (exact on all rows)
#   dmean  = floor(dbytes / dpkts + 0.5), 0 if dpkts==0 (exact on all rows)
#   rate   = (spkts + dpkts - 1) / dur  when dur > 0    (relative error < 3e-5)
# rate when dur == 0 cannot be derived (real values are 0 to about 0.09),
# so it is resampled from the real dur == 0 rows.

def rebuild_unsw(syn, normal, rng):
    syn = syn.copy()

    syn["tcprtt"] = syn["synack"] + syn["ackdat"]

    s_pos = syn["spkts"] > 0
    syn["smean"] = np.where(s_pos, np.floor(syn["sbytes"] / syn["spkts"].where(s_pos, 1) + 0.5), 0)

    d_pos = syn["dpkts"] > 0
    syn["dmean"] = np.where(d_pos, np.floor(syn["dbytes"] / syn["dpkts"].where(d_pos, 1) + 0.5), 0)

    dur = syn["dur"].to_numpy(dtype=float)
    dur_pos = dur > 0
    packets = (syn["spkts"] + syn["dpkts"] - 1).to_numpy(dtype=float)
    rate = np.zeros(len(syn))
    rate[dur_pos] = packets[dur_pos] / dur[dur_pos]

    zero_pool = normal.loc[normal["dur"] == 0, "rate"].to_numpy()
    n_zero = int((~dur_pos).sum())
    if n_zero and len(zero_pool):
        rate[~dur_pos] = rng.choice(zero_pool, size=n_zero)
    syn["rate"] = rate

    return syn


DERIVED_BUILDERS = {
    "unsw": rebuild_unsw,
    # "nslkdd": no derived columns
}


# ============================================================
# 4. Post-processing
# ============================================================

def snap_to_observed(values, observed):
    """Replace each value with the nearest value that appears in the real data."""
    obs = np.sort(np.unique(observed))
    idx = np.clip(np.searchsorted(obs, values), 1, len(obs) - 1)
    left, right = obs[idx - 1], obs[idx]
    return np.where(np.abs(values - left) <= np.abs(values - right), left, right)


def derived_in_range(rows, info):
    """Booleans per derived column: is the rebuilt value inside the real normal range?"""
    normal = info["normal"]
    ok = pd.DataFrame(index=rows.index)
    for col in info["derived"]:
        ok[col] = (np.isfinite(rows[col])
                   & (rows[col] >= normal[col].min() - 1e-9)
                   & (rows[col] <= normal[col].max() + 1e-9))
    return ok


def fit_purifier(train_csv, info):
    """
    Label-informed filter (Change D). Trains a classifier on the real labelled rows
    (normal = 0, anomaly = 1) and returns score(df) -> probability that each row of df
    (feature columns only) looks like an anomaly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    train = pd.read_csv(train_csv)
    feats, cats = info["feature_columns"], info["types"]["categorical"]
    X = pd.get_dummies(train[feats], columns=cats, dtype=float)
    cols = X.columns
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_SEED)
    clf.fit(X, train[LABEL_COLUMN])

    def score(df):
        Z = pd.get_dummies(df[feats], columns=cats, dtype=float)
        return clf.predict_proba(Z.reindex(columns=cols, fill_value=0.0))[:, 1]

    return score


def lift_tails(syn, info):
    """
    Lift the upper tail of the columns in info["tail_fits"] (Change C).
    A synthetic row at rank fraction p above the real fraction f0 (the threshold) is
    placed at tail position t = (p - f0) / (1 - f0) and gets at least the real value at
    that position, interpolated between the stored tail quantiles (linear in log1p over
    the depth -log2(1 - t)). Values are never lowered.
    """
    normal, types = info["normal"], info["types"]
    syn = syn.copy()

    for col, fit in info["tail_fits"].items():
        v = syn[col].to_numpy(dtype=float)
        p = (pd.Series(v).rank(method="average").to_numpy() - 0.5) / len(v)
        tail = p > fit["f0"]
        t = (p[tail] - fit["f0"]) / (1 - fit["f0"])
        depth = -np.log2(1 - t)
        mapped = np.expm1(np.interp(depth, fit["depth"], fit["log_values"]))
        v[tail] = np.maximum(v[tail], mapped)
        syn[col] = v

    # keep bytes and packets consistent: more bytes need enough packets for a plausible mean
    # packet size (the mean-size columns themselves are rebuilt later)
    for byte_col, (pkt_col, mean_col) in DATASETS[info["name"]].get("byte_caps", {}).items():
        if byte_col in info["tail_fits"]:
            mean_max = float(normal[mean_col].max())
            pkts = syn[pkt_col].to_numpy(dtype=float)
            need = np.ceil(syn[byte_col].to_numpy(dtype=float) / mean_max)
            pkts = np.maximum(pkts, np.minimum(need, float(normal[pkt_col].max())))
            syn[pkt_col] = pkts
            syn[byte_col] = np.minimum(syn[byte_col].to_numpy(dtype=float), pkts * mean_max)

    # same final clean-up as the other columns of that type
    for col in {c for c in info["tail_fits"]} | {p for b in DATASETS[info["name"]].get("byte_caps", {}).values() for p in b[:1]}:
        lo, hi = normal[col].min(), normal[col].max()
        syn[col] = syn[col].astype(float).clip(lo, hi)
        if col in types["integer"]:
            syn[col] = np.rint(syn[col])
        elif col in types["decimals"]:
            syn[col] = syn[col].round(types["decimals"][col])
    return syn


def postprocess(raw, info, rng):
    """
    Turn raw model output (model_columns only) into full feature rows.
    Returns (rows, keep) where keep marks rows whose rebuilt (derived) columns
    stay inside the real normal range. The caller should oversample and keep
    only rows where keep is True.
    """
    normal, types, derived = info["normal"], info["types"], info["derived"]
    feature_columns = info["feature_columns"]
    syn = raw[info["model_columns"]].copy()
    n = len(syn)

    # undo log1p; clip in log space first so expm1 cannot overflow
    for col, scale in info["log_columns"].items():
        top = np.log1p(float(normal[col].max()) * scale)
        syn[col] = np.expm1(syn[col].astype(float).clip(0, top)) / scale

    # constant columns go back unchanged
    for col, value in types["constant"].items():
        syn[col] = value

    # derived columns are not generated by the model; they are rebuilt further down
    not_derived = lambda cols: [c for c in cols if c not in derived]

    # discrete numeric and binary columns: only values seen in training
    for col in not_derived(types["binary"] + types["discrete"]):
        syn[col] = snap_to_observed(syn[col].astype(float).to_numpy(), normal[col].to_numpy())

    # integer columns: clip to observed range, whole numbers
    for col in not_derived(types["integer"]):
        lo, hi = normal[col].min(), normal[col].max()
        syn[col] = np.rint(syn[col].astype(float).clip(lo, hi))

    # continuous columns: clip to observed range, keep the real rounding precision
    for col in not_derived(types["continuous"]):
        syn[col] = syn[col].astype(float).clip(normal[col].min(), normal[col].max())
        if col in types["decimals"]:
            syn[col] = syn[col].round(types["decimals"][col])

    # zero flags: flagged rows get exact zeros in the listed columns
    for flag in info["zero_flags"]:
        mask = raw[flag["name"]].astype(float).to_numpy() >= 0.5
        for col in flag["zero"]:
            syn.loc[mask, col] = 0

    # thin upper tails are lifted after the zeros are in place (ranks include the zeros)
    if info["tail_fits"]:
        syn = lift_tails(syn, info)

    # derived columns come from their generated source columns
    keep = np.ones(n, dtype=bool)
    if derived:
        syn = DERIVED_BUILDERS[info["name"]](syn, normal, rng)
        keep &= derived_in_range(syn, info).all(axis=1).to_numpy()

    # original column order (dtypes are restored in finalize, after discarding rows)
    return syn[feature_columns], keep


# ============================================================
# 5. Validate and finalize
# ============================================================

def validate(sub, info, target_n):
    """Submission rules from the starter kit, plus category and discrete-value checks."""
    feature_columns, normal, types = info["feature_columns"], info["normal"], info["types"]

    if len(sub) != target_n:
        raise ValueError(f"Submission must contain exactly {target_n} rows.")

    if sub.columns.tolist() != ["id"] + feature_columns:
        raise ValueError("Submission columns are incorrect.")

    if sub.isna().any().any():
        raise ValueError("Submission contains missing values.")

    numeric_values = sub.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Submission contains NaN or infinite values.")

    for col in types["categorical"]:
        unseen = set(sub[col].unique()) - set(normal[col].unique())
        if unseen:
            raise ValueError(f"Column '{col}' has categories not seen in training: {unseen}")

    for col in types["binary"] + types["discrete"]:
        unseen = set(sub[col].unique()) - set(normal[col].unique())
        if unseen:
            raise ValueError(f"Column '{col}' has values not seen in training: {sorted(unseen)}")

    print("Validation passed.")


def finalize(syn, keep, info, target_n):
    """Keep valid rows, cut to target_n, restore dtypes, add the id column, validate."""
    normal = info["normal"]
    kept = syn[keep].reset_index(drop=True)
    if len(kept) < target_n:
        raise ValueError(
            f"Only {len(kept)} of {len(syn)} sampled rows passed the range check; "
            f"need {target_n}. Sample more rows."
        )
    sub = kept.iloc[:target_n].copy()

    # integer columns stay integers (done here because discarded rows may hold inf/NaN)
    for col in info["feature_columns"]:
        if pd.api.types.is_integer_dtype(normal[col]):
            sub[col] = np.rint(sub[col]).astype(normal[col].dtype)

    sub.insert(0, "id", np.arange(target_n))
    validate(sub, info, target_n)
    return sub


# ============================================================
# 6. Self-test (no model): real normal rows -> drop -> rebuild -> compare
# ============================================================

def selftest(name, train_csv, seed=RANDOM_SEED):
    info = prepare(name, train_csv)
    normal, types = info["normal"], info["types"]
    rng = np.random.default_rng(seed)

    # pretend the model reproduced the real rows exactly
    raw = training_frame(info)
    lifted = dict(info, tail_fits=info["tail_fits"])
    info = dict(info, tail_fits={})          # exact round trip is checked without the lift
    rebuilt, keep = postprocess(raw, info, rng)

    print("\n--- Round-trip results ---")
    print(f"Rows kept by range check: {keep.sum()} / {len(keep)}")

    # columns the model saw should come back unchanged
    changed = {}
    for col in info["model_columns"]:
        a, b = normal.loc[keep, col], rebuilt.loc[keep, col]
        same = (a == b) if col in types["categorical"] else np.isclose(
            a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9)
        if not same.all():
            changed[col] = float((~same).mean())
    print("Model columns changed by post-processing:",
          changed if changed else "none")

    # derived columns should be reproduced from their sources
    for col in info["derived"]:
        a, b = normal.loc[keep, col].astype(float), rebuilt.loc[keep, col].astype(float)
        mask = np.ones(len(a), dtype=bool)
        if col == "rate":  # dur == 0 rows are resampled, not derived
            mask = (normal.loc[keep, "dur"] > 0).to_numpy()
        match = np.isclose(a[mask], b[mask], rtol=1e-4, atol=1e-9)
        print(f"Derived '{col}': {match.mean():.4%} reproduced "
              f"(checked {mask.sum()} rows)")

    # the whole pipeline end to end
    sub = finalize(rebuilt, keep, info, int(keep.sum()))
    print(f"Finalize OK: {sub.shape}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Profile and self-test the data processing.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--train", type=Path, help="Training CSV (with is_anomaly).")
    args = parser.parse_args()

    selftest(args.dataset, args.train or DATASETS[args.dataset]["train"])


if __name__ == "__main__":
    main()
