# CTGAN route: normal-only synthetic data for NSL-KDD and UNSW-NB15 (Group 13)

CS5344 project. We submit synthetic **normal** rows (40,000 for NSL-KDD, 70,000 for UNSW-NB15).
The organisers train ECOD and Isolation Forest on our rows only and score hidden rows with AUPRC.
This folder holds the CTGAN-based generator and the column handling around it. It is separate
from `adi's pipeline/` (conditional VAE) and does not change it.

The folder is meant to sit next to `data/` in the repo, so `../data/train_*.csv` points at the released
training files.

## Files

| File | Purpose |
|---|---|
| `generate.py` | Trains CTGAN, samples rows, optional label-informed filter (`--purify`), writes the submission CSV |
| `data_processing.py` | Column types, log-transform of heavy tails, zero flags, tail lift, derived-column rebuild, range check, validation |
| `requirements.txt` | Python packages |
| `pipeline3_ctgan.py` | Adi's `pipeline3.py` with three small edits that let it use our CTGAN improvements (see the last section) |

`generate.py` and `data_processing.py` are a matched pair. Always update them together.

## How it works

1. **Load and profile** (`prepare`): keep the normal rows, find categorical, constant, binary, discrete
   (<= 15 distinct values), integer and continuous columns. Constant and derived columns are not modelled.
2. **Train CTGAN** on the remaining columns, with categorical, binary and discrete columns declared discrete.
3. **Sample and post-process** (`postprocess`): undo the log transform, snap discrete values to ones seen
   in training, clip and round, rebuild derived columns, drop rows outside the real normal range, and
   sample more until the target count is reached.
4. **Purify (UNSW only)**: sample 70,000 / 0.8 valid rows, score each with a gradient-boosted classifier
   trained on the labelled rows, and keep the 70,000 that look most normal.
5. **Validate** the schema (`id` plus feature columns, no NaN, no unseen categories) and save.

## What we changed, in order

| Change | What it does | Effect (public AUPRC) |
|---|---|---|
| Baseline | Plain CTGAN | UNSW about 0.36 |
| `log_frequency=False` (UNSW only) | Pick the conditioning category by its real frequency, not log-flattened. Stops rare categories (`arp`, `sttl` = 0) being over-produced | UNSW about 0.42 |
| Log transform | Train heavy-tailed columns as `log1p(x)` and invert after sampling | NSL-KDD 0.92561 (from about 0.91); UNSW 0.41286 |
| **A** | UNSW sub-second columns (`dur`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `synack`, `ackdat`) are scaled by 1000 before `log1p`, because `log1p` barely changes values below 1 | UNSW 0.41981 |
| **B** | Zero flags: a 0/1 column marks "no reply" (`dpkts` = 0) and "`dur` = 0" rows, and the linked columns are set to exact zeros | UNSW 0.43292 (0.42530 with seed 7); NSL-KDD 0.92116, worse, so switched off |
| **C** | Tail lift: CTGAN's upper tails were too thin (for example `dbytes` p99 at 6% of the real value). The top 5% of synthetic values is lifted to the real tail quantiles (lift only; bytes are capped by packets times the real maximum mean size) | UNSW 0.445 (0.43064 with seed 7) |
| **D** | `--purify KEEP`: keep the most normal-looking fraction of the sampled rows, as scored by the classifier above | UNSW C + 90% 0.45980; **C + 80% 0.47896 (final)** |
| D, within group | `--purify-within sttl ct_state_ttl` keeps the same share in each value of those columns | UNSW 0.44492, worse |

All scores are public-leaderboard AUPRC from 100-epoch runs. Rerun noise on UNSW is about +/-0.01, so
differences of that size are not conclusive. The seed-7 results show this.

Each change is a switch in the final files, so any of them can be reproduced:

| To reproduce | Flags |
|---|---|
| A | `--no-zero-flags --no-tail-lift` |
| B | `--no-tail-lift` |
| C | none |
| D (final UNSW setup) | `--purify 0.8` |

NSL-KDD zero flags are switched off in `DATASETS`; the commented-out block shows how to turn them back on.

## Final submitted setup

| | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Log transform | 9 columns | 9 columns, plus 7 sub-second columns scaled by 1000 |
| Zero flags | off | on (`no_reply`, `dur_zero`) |
| Tail lift | n/a | on (11 columns) |
| `log_frequency` | default | False |
| Purify | off | keep 80% |
| Epochs, seed | 100, 42 | 100, 42 |
| Public AUPRC | 0.92561 | 0.47896 |

For NSL-KDD the log transform alone was best; flags scored 0.92116.

## Run it

Run from this folder, with the training files in `../data/`.

```bash
pip install -r requirements.txt

# NSL-KDD  (40,000 rows)
python generate.py --dataset nslkdd --train ../data/train_NSL-KDD.csv --epochs 100 \
    --out outputs/synthetic_nsl-kdd.csv --model-out models/ctgan_nslkdd.pt

# UNSW-NB15  (70,000 rows, final setup: tail lift + purify 80%)
python generate.py --dataset unsw --train ../data/train_UNSW-NB15.csv --epochs 100 --purify 0.8 \
    --out outputs/synthetic_unsw-nb15.csv --model-out models/ctgan_unsw.pt
```

Useful options:

- `--load-model models/ctgan_unsw.pt`: skip training and resample from a saved model (about 6 seconds for 70,000 rows). Use the same `--no-zero-flags` and `--no-tail-lift` settings that the model was trained with.
- `--purify-within COLUMN ...`: with `--purify`, keep the mix of those columns.
- `--seed N`: change the random seed (default 42).
- Smoke test: `python generate.py --dataset unsw --train ../data/train_UNSW-NB15.csv --epochs 2 --n 2000 --out smoke.csv`
- Self-test of the column handling, no model needed: `python data_processing.py --dataset unsw --train ../data/train_UNSW-NB15.csv`

Only the released `train` files are used; validation, test and Kaggle rows are never read. Models and
outputs are not committed.

## Using this with Adi's pipeline

`pipeline3_ctgan.py` is `pipeline3.py` with three edits: the CTGAN epoch setting, an optional log
transform for heavy-tailed columns (bounded 0 to 1 columns are skipped), and `log_frequency` turned off.
A switch controls them:

```bash
P3_CTGAN_OURS=1   # our CTGAN changes (default in this file)
P3_CTGAN_OURS=0   # Adi's original CTGAN settings, for comparison
```

To run it for UNSW with the CTGAN generator (the script reads the two training CSVs from the working directory):

```bash
cp ../data/train_NSL-KDD.csv ../data/train_UNSW-NB15.csv .
P3_DATASETS=UNSW-NB15 P3_GENERATOR="UNSW-NB15=ctgan" python pipeline3_ctgan.py
```

Output goes to `final_pipeline3_output_group13/` (set `P3_OUT` to change it). With our changes in that
pipeline, UNSW scored about 0.52. Adi's own VAE route scores 0.5882 (UNSW) and 0.9294 (NSL-KDD).

## Known limits

- `rebuild_unsw` in `data_processing.py` holds the UNSW derived-column formulas (`tcprtt`, `smean`,
  `dmean`, `rate`), verified on the real normal rows. It is registered in `DERIVED_BUILDERS`; NSL-KDD has
  no derived columns.
- The purify step uses the training labels. It lifted the public score from 0.445 to 0.479, but it works
  partly by shifting `sttl` and `ct_state_ttl`, so the gain may not carry over to attack types that are
  unseen in training. Fixing the mix of those columns (`--purify-within`) lost the gain.
- Reruns on a different machine can move UNSW by about +/-0.01.
