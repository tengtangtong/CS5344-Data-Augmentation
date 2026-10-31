# Adi's pipeline — normal-only synthetic data for fixed ECOD + IForest

Final code for the CS5344 augmentation task (NSL-KDD and UNSW-NB15). **One pipeline,
`pipeline3.py`, produces both submissions**; only a few per-dataset hyperparameters differ.

| Dataset | Kaggle public AUPRC | File |
|---|---|---|
| NSL-KDD | **0.92960** | `pipeline3_output/submission_NSL-KDD_p3.csv` (40,000 rows) |
| UNSW-NB15 | **0.57760** | `pipeline3_output/submission_UNSW-NB15_p3.csv` (70,000 rows) |

The NSL-KDD file is the 0.92960 file with its 9 duplicate rows / copies of real normals replaced
(see the Dedupe step below). The UNSW-NB15 output is byte-identical (SHA-256) to the file that
scored 0.57760.

## Quick start

```bash
pip install -r requirements.txt
cp ../data/train_NSL-KDD.csv ../data/train_UNSW-NB15.csv .   # or keep them in the working directory
python pipeline3.py                    # both datasets → pipeline3_output/   (≈ 10–15 min per dataset, 2-core CPU)
P3_DATASETS=UNSW-NB15 python pipeline3.py
P3_MODE=final+evaluate python pipeline3.py   # + Pipeline 1 vs 3 on held-out attack types
P3_FAST=1 python pipeline3.py         # smoke test, ≈ 8 min for both (smaller learning; not the submission)
```

Environment options: `P3_MODE` (`final` | `evaluate` | `final+evaluate`), `P3_DATASETS`,
`P3_OUT` (output folder), `P3_SCORER` (trimming critic override, e.g. `clf`, `lr`, `clf+lr`),
`P3_CRITIC` (`default` | `strong`), `P3_ECOD` (`pyod` exact rule, default | `old`), `P3_FAST`.
Requires only NumPy, pandas, SciPy and scikit-learn (Python ≥ 3.8). Everything is seeded and
reruns are byte-identical. Validation, test and Kaggle rows are never used: all learning happens on
splits of the released `train.csv`.

## The pipeline and why each step is there

```
released train.csv (normal + seen anomalies)
  ├─ 1. Curation        ENN: drop normals whose 10 nearest labelled neighbours are mostly attacks
  ├─ 2. Derived flow    remove verified identities / constant columns, restore after generation
  ├─ 3. Anomaly critic  HistGB P(anomaly) (+ logistic regression on UNSW); UNSW: leave-one-type-out
  ├─ 4. Generator       mixed-type conditional VAE (NumPy), conditioned on the category tuple
  ├─ 5. Over-generate   ×20 candidates, per-tuple quotas from (1 − P)^8 weights
  ├─ 6. Trim 90%        keep the least attack-like 10% per tuple
  ├─ 7. Sculpt          sampling-importance-resampling to the weighted real density
  ├─ 8. Column shaping  UNSW: per-column plan (off / κ 4 / κ 50 / two-point) learned on behaviour
  │                     splits with an unseen-type penalty, scored with the exact PyOD ECOD rule
  ├─ 9. Dedupe          replace duplicate rows / copies of real normals with unused candidates
  └─10. Validate        schema, no NaN, no invented categories, values inside the real-normal range
```

| Step | Problem it solves | Why it helps ECOD and IForest | Evidence |
|---|---|---|---|
| ENN curation | Mislabelled / boundary normals teach the generator attack regions | Fewer synthetic rows where attacks live → attacks fall in tails and sparse cells | Standard editing rule (Wilson, 1972) |
| Derived flow | Generators break identities such as `smean = sbytes / spkts` | No normal mass in impossible places | Identities hold on ≥ 95% of real normals |
| Conditional VAE | Need diverse candidates with real category tuples | Categoricals create no false tails | Every learned generator beat the baseline by > 0.10 |
| Over-generate ×20 + trim 90% | UNSW normals overlap attacks (`sttl` = 254 in 27%) | Attack-shared modes become tails ECOD flags and regions IForest isolates | +0.092 Kaggle (with sculpt) |
| Sculpt | Trimming over-concentrates mass | Keeps rarer legitimate normals → fewer false positives | Removing it hurt in combination |
| Range-spreading κ = 50 | IForest wastes random splits on wide columns with no signal | Near-constant body blinds IForest; thin tail to the real max keeps ECOD calm | +0.036 Kaggle |
| Learned per-column plan | One κ for all columns; tuning on known types overfits | Shape both detectors accept, gains on seen types only penalised | +0.011 Kaggle |
| Behaviour-based held-out types | Need a stand-in for unseen attack types | Tests whether a change helps types it was not tuned on | Seen-type probes lost up to 0.013 |
| Exact PyOD ECOD rule | Local score must rank files like Kaggle | Optimise for the detector actually used | Spearman ρ −0.19 → +0.83 |
| HistGB + logistic regression critic | A tree critic keys on seen-type thresholds | Two models that err differently must both accept a candidate | +0.010 held-out (local); +0.0035 Kaggle |
| Dedupe | Repeated rows / copies of real rows | No information for IForest, extra ties for ECOD, not synthetic | 9 NSL rows replaced; UNSW none |

## Per-dataset hyperparameters

| Setting | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Generator | conditional VAE | conditional VAE |
| Anomaly model | standard 5-fold HistGB | leave-one-type-out (behaviour types) |
| Trimming critic (`scorer`) | HistGB P(anomaly) + realism | HistGB + logistic regression, rank-averaged |
| Default shaping (Pipeline 1) | none | κ = 50, δ = 0.10, 7 columns |
| `learn_shaping` | off | on |
| `dedupe` | on | on |
| Seed | 2026 | 2027 |

NSL-KDD normals are already well separated (0.93 with real normals), and every reshaping we tried
lowered its public score, so shaping is off there. UNSW-NB15 normals overlap attacks, so it gets
learned shaping and the second critic. Settings live in `PER_DATASET` (Pipeline 1 part) and
`PER_DATASET_3` (Pipeline 2/3 part) inside `pipeline3.py`.

## Kaggle public scores behind the final settings (UNSW-NB15)

| Milestone | Public AUPRC |
|---|---|
| Tuple donor-seed baseline | 0.43321 |
| VAE + trim + sculpt (Pipeline 0) | 0.52515 |
| + range-spread 7 columns, κ = 2 | 0.55045 |
| κ = 50 (same pool) | 0.56135 |
| Pipeline 1 final | 0.56335 |
| Pipeline 2, exact ECOD rule, learned plan | 0.57409 |
| **Pipeline 3, HistGB + logistic-regression critic** | **0.57760** |

Not adopted (each tested, see `exp_*.py`): regularised HistGB critic (0.57182 UNSW, 0.92937 NSL),
three probes tuned to known attack types (0.55279 / 0.54161 / 0.55386), category reshaping,
decorrelating columns, discretising values (in-sample UNSW 0.582 → 0.545–0.581), unsupervised critics
(autoencoder, kNN) alone. Run-to-run noise on the public board is about ±0.005.

## Files

| File | Purpose |
|---|---|
| **`pipeline3.py`** | **Final single-file pipeline** (Pipeline 1 stages + Pipeline 2 learning + Pipeline 3 critic and dedupe) |
| `pipeline1.py` | Over-generate → trim → sculpt → range-spread; also the generator comparison and ablation modes |
| `pipeline2.py` | Imports Pipeline 1; the same learning steps as `pipeline3.py` in modular form (used by the experiments) |
| `exp_critics2.py` | Trimming-critic screen (HistGB, LR, RF, ET, MLP, NB, kNN, autoencoder, pairs) on behaviour splits |
| `exp_critics.py` | First critic screen (autoencoder, MLP, LR partners) |
| `exp_regularize.py` | Regularised HistGB critic variants |
| `exp_decorrelate.py` | Column permutation (breaking correlations) |
| `exp_discretize.py` | Snapping / quantile binning of the final files (in-sample, exact ECOD + IForest) |
| `exp_categories.py` | Category-tuple reweighting |
| `exp_twopoint.py` | Two-point neutralising under the old ECOD rule |
| `check_ecod_rule.py` | Old vs exact ECOD rule against Kaggle-scored files (needs those CSVs locally) |
| `p2_robustness.py` | Scores a learned plan on further held-out splits |
| `gen_overfit.py` | The three known-type probes (negative result) |
| `results/` | Metadata of the final runs: settings, learned plans, SHA-256, dedupe counts |

Most experiments take a dataset name and split rotations, e.g. `python exp_critics2.py UNSW-NB15 0 1 2`,
and expect the training CSVs next to the scripts or in the working directory.
