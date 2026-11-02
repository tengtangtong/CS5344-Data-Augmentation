"""
CS5344 - Synthetic NORMAL data generation with CTGAN (Group 13).

Column handling (types, constant and derived columns, clipping, validation)
lives in data_processing.py. This file only trains CTGAN and samples from it.

Pipeline (one dataset per run):
1. prepare(): load normal rows, detect column types, choose model columns.
2. Fit CTGAN on the model columns (discrete columns declared).
3. Sample rows; post-process; discard rows whose rebuilt columns leave the
   real normal range; sample more until target_n valid rows exist.
4. Validate and save id + feature columns.

Usage
-----
    python generate.py --dataset unsw   --train path/to/UNSW-NB15/train.csv
    python generate.py --dataset nslkdd --train path/to/NSL-KDD/train.csv

Quick smoke test (2 epochs, 2000 rows):
    python generate.py --dataset unsw --train train.csv --epochs 2 --n 2000 --out smoke.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_processing as dp


# ============================================================
# Configuration
# ============================================================

HERE = Path(__file__).resolve().parent
RANDOM_SEED = dp.RANDOM_SEED

CTGAN_PARAMS = {
    "epochs": 300,
    "batch_size": 500,        # must be divisible by pac
    "pac": 10,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256),
}

# Per-dataset overrides on top of CTGAN_PARAMS.
# log_frequency=False: during training, pick the conditioning category with its
# real frequency instead of a log-flattened one. UNSW over-produced rare
# categories (arp, sttl=0, ...) with the default True, so it is tested there only.
DATASET_PARAMS = {
    "unsw": {"log_frequency": False},
    "nslkdd": {},
}

OVERSAMPLE = 1.2    # safety margin on top of the observed keep rate
MAX_ROUNDS = 50      # give up if rows keep failing the range check
MIN_KEEP_RATE = 0.02 # lowest keep rate assumed when sizing a round
MAX_DRAW = 500_000   # cap on rows sampled in one round (memory)


# ============================================================
# 1. Train CTGAN
# ============================================================

def fit_ctgan(train_df, discrete_columns, seed, params):
    # imported here so the file can be read without torch installed
    import torch
    from ctgan import CTGAN

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = CTGAN(verbose=True, **params)
    if hasattr(model, "set_random_state"):
        model.set_random_state(seed)
    model.fit(train_df, discrete_columns=discrete_columns)
    return model


# ============================================================
# 2. Sample until enough valid rows exist
# ============================================================

def sample_valid(model, info, target_n, rng):
    """Sample, post-process and keep only rows that pass the range check."""
    kept, n_kept = [], 0
    keep_rate = 0.7   # first guess; replaced by the observed rate after round 1

    for round_no in range(1, MAX_ROUNDS + 1):
        need = target_n - n_kept
        draw = int(np.ceil(need * OVERSAMPLE / max(keep_rate, MIN_KEEP_RATE)))
        draw = min(draw, MAX_DRAW)
        raw = model.sample(draw)
        rows, keep = dp.postprocess(raw, info, rng)

        discarded = int((~keep).sum())
        keep_rate = float(keep.mean())
        print(f"Round {round_no}: sampled {draw}, discarded {discarded} "
              f"({discarded / draw:.1%}), kept {int(keep.sum())}")
        if info["derived"]:
            fails = (~dp.derived_in_range(rows, info)).mean()
            print("  failed range check by column: "
                  + ", ".join(f"{c} {v:.1%}" for c, v in fails.items()))

        kept.append(rows[keep])
        n_kept += int(keep.sum())
        if n_kept >= target_n:
            break
    else:
        raise ValueError(f"Only {n_kept} of {target_n} valid rows after {MAX_ROUNDS} rounds.")

    rows = pd.concat(kept, ignore_index=True)
    return rows, np.ones(len(rows), dtype=bool)


# ============================================================
# 3. Run one dataset
# ============================================================

def run(name, train_csv, out_csv, target_n, seed, params, model_path, load_model=None,
        use_zero_flags=True, use_tails=True, purify=None, purify_within=None):
    rng = np.random.default_rng(seed)
    info = dp.prepare(name, train_csv, use_zero_flags, use_tails)

    if load_model:
        # skip training: sample from a model saved by an earlier run
        from ctgan import CTGAN
        model = CTGAN.load(str(load_model))
        print(f"Loaded model: {load_model}")
    else:
        model = fit_ctgan(dp.training_frame(info),
                          info["discrete_columns"], seed, params)
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            print(f"Saved model: {model_path}")

    if purify:
        # Change D: draw target_n / purify valid rows, keep the target_n that the
        # anomaly classifier finds most normal-looking.
        need = int(np.ceil(target_n / purify))
        rows, keep = sample_valid(model, info, need, rng)
        rows = rows.iloc[:need].reset_index(drop=True)
        scores = dp.fit_purifier(train_csv, info)(rows)
        if purify_within:
            # rank rows inside each group of the given column(s), so every group keeps
            # about the same fraction and the mix of group values is preserved
            within = pd.Series(scores).groupby([rows[c] for c in purify_within]).rank(
                method="first", pct=True).to_numpy()
            order = np.lexsort((scores, within))      # by within-group rank, then score
        else:
            order = np.argsort(scores, kind="stable")
        chosen = np.sort(order[:target_n])
        dropped = np.setdiff1d(np.arange(len(rows)), chosen)
        if purify_within:
            for c in purify_within:
                before = rows[c].value_counts(normalize=True)
                after = rows.iloc[chosen][c].value_counts(normalize=True)
                print(f"  share of '{c}' before -> after: " + ", ".join(
                    f"{v}: {before[v]:.1%}->{after.get(v, 0):.1%}" for v in before.index[:5]))
        print(f"Purify: kept the {target_n} most normal-looking of {len(rows)} rows "
              f"(score cut-off {scores[chosen].max():.3f}; mean score kept "
              f"{scores[chosen].mean():.3f}, dropped {scores[dropped].mean():.3f})")
        rows = rows.iloc[chosen].reset_index(drop=True)
        keep = np.ones(len(rows), dtype=bool)
    else:
        rows, keep = sample_valid(model, info, target_n, rng)
    sub = dp.finalize(rows, keep, info, target_n)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} {sub.shape}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic normal data with CTGAN.")
    parser.add_argument("--dataset", required=True, choices=sorted(dp.DATASETS))
    parser.add_argument("--train", type=Path, help="Training CSV (with is_anomaly).")
    parser.add_argument("--out", type=Path, help="Output CSV path.")
    parser.add_argument("--n", type=int, help="Number of synthetic rows.")
    parser.add_argument("--epochs", type=int, help="CTGAN epochs (default 300).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model-out", type=Path, help="Where to save the trained CTGAN.")
    parser.add_argument("--load-model", type=Path,
                        help="Skip training and sample from this saved CTGAN (.pt).")
    parser.add_argument("--no-zero-flags", action="store_true",
                        help="Disable Change B (zero flags) to reproduce the earlier pipeline. "
                             "Use the same setting when loading a saved model.")
    parser.add_argument("--no-tail-lift", action="store_true",
                        help="Disable Change C (tail lift). Use the same setting when "
                             "loading a saved model.")
    parser.add_argument("--purify", type=float, metavar="KEEP",
                        help="Change D: keep this fraction (0-1) of the most normal-looking "
                             "rows, e.g. 0.9. Needs the labelled training CSV.")
    parser.add_argument("--purify-within", nargs="+", metavar="COLUMN",
                        help="With --purify: filter inside each value of these columns, so "
                             "their value mix does not change (e.g. --purify-within sttl).")
    args = parser.parse_args()
    if args.purify_within and args.purify is None:
        parser.error("--purify-within needs --purify.")
    if args.purify is not None and not 0 < args.purify <= 1:
        parser.error("--purify must be in (0, 1].")

    preset = dp.DATASETS[args.dataset]
    params = dict(CTGAN_PARAMS)
    params.update(DATASET_PARAMS.get(args.dataset, {}))
    if args.epochs:
        params["epochs"] = args.epochs

    run(
        name=args.dataset,
        train_csv=args.train or preset["train"],
        out_csv=args.out or preset["out"],
        target_n=args.n or preset["n"],
        seed=args.seed,
        params=params,
        model_path=args.model_out or HERE / "models" / f"ctgan_{args.dataset}.pt",
        load_model=args.load_model,
        use_zero_flags=not args.no_zero_flags,
        use_tails=not args.no_tail_lift,
        purify=args.purify,
        purify_within=args.purify_within,
    )


if __name__ == "__main__":
    main()
