# Adi's pipeline -- normal-only synthetic data for fixed ECOD + IForest

Final code for the CS5344 augmentation task (NSL-KDD and UNSW-NB15). **One pipeline,
`pipeline3.py`, produces both submissions**; only a few per-dataset hyperparameters differ.

| Dataset | Kaggle public AUPRC | File |
|---|---|---|
| NSL-KDD | **0.92935** | `pipeline3_output/submission_NSL-KDD_p3.csv` (40,000 rows) |
| UNSW-NB15 | **0.58821** | `pipeline3_output/submission_UNSW-NB15_p3.csv` (70,000 rows) |

Reruns are byte-identical on the same machine. Across machines, floating-point differences (CPU, BLAS,
library versions) change the generated rows, so the UNSW-NB15 column plan learned in the 0.57760 run is
stored in the code (`UNSW_PLAN_0_57760`) and applied rather than re-learned, and the untested joint row
swap is off. With these settings a second machine scored 0.92935 / 0.58821; re-learning the plan there
(`P3_RELEARN=1`, with row swapping on) gave 0.56936. Expect UNSW-NB15 reruns to vary by about +/-0.01.

## Quick start

```bash
pip install -r requirements.txt
cp ../data/train_NSL-KDD.csv ../data/train_UNSW-NB15.csv .   # or keep them in the working directory
python pipeline3.py                    # both datasets -> pipeline3_output/   (~ 10-15 min per dataset, 2-core CPU)
P3_DATASETS=UNSW-NB15 python pipeline3.py
P3_MODE=final+evaluate python pipeline3.py   # + Pipeline 1 vs 3 on held-out attack types
P3_FAST=1 python pipeline3.py         # smoke test, ~ 8 min for both (smaller learning; not the submission)
```

Environment options: `P3_MODE` (`final` | `evaluate` | `final+evaluate`), `P3_DATASETS`,
`P3_OUT` (output folder), `P3_SCORER` (trimming critic override, e.g. `clf`, `lr`, `clf+lr`),
`P3_CRITIC` (`default` | `strong`), `P3_ECOD` (`pyod` exact rule, default | `old`), `P3_FAST`,
`P3_RELEARN=1` (re-learn the UNSW plan), `P3_JOINT=1` (joint row swap),
`P3_GENERATOR` (generator per dataset, e.g. `P3_GENERATOR="NSL-KDD=baseline,UNSW-NB15=vae"`; options `vae`, `baseline`, `gmm`, `ctgan`).
Requires only NumPy, pandas, SciPy and scikit-learn (Python >= 3.8). `pyod` is optional and does not change the output (the exact ECOD rule is built in); `ctgan` is needed only for `generator="ctgan"`. Everything is seeded, and
reruns are byte-identical on the same machine. Validation, test and Kaggle rows are never used: all learning happens on
splits of the released `train.csv`.

## The pipeline and why each step is there

```
released train.csv (normal + seen anomalies)
  |- 1. Curation        ENN: drop normals whose 10 nearest labelled neighbours are mostly attacks
  |- 2. Derived flow    remove verified identities / constant columns, restore after generation
  |- 3. Anomaly critic  HistGB P(anomaly) (+ logistic regression on UNSW); UNSW: leave-one-type-out
  |- 4. Generator       mixed-type conditional VAE (NumPy), conditioned on the category tuple
  |- 5. Over-generate   x20 candidates, per-tuple quotas from (1 - P)^8 weights
  |- 6. Trim 90%        keep the least attack-like 10% per tuple
  |- 7. Sculpt          sampling-importance-resampling to the weighted real density
  |- 8. Column shaping  UNSW: per-column plan (off / kappa 4 / kappa 50 / two-point) learned once on behaviour
  |                     splits with an unseen-type penalty (exact PyOD ECOD rule), then fixed in the code
  |- 9. Dedupe          replace duplicate rows / copies of real normals with unused candidates
  `-10. Validate        schema, no NaN, no invented categories, values inside the real-normal range
```

| Step | Problem it solves | Why it helps ECOD and IForest | Evidence |
|---|---|---|---|
| ENN curation | Mislabelled / boundary normals teach the generator attack regions | Fewer synthetic rows where attacks live -> attacks fall in tails and sparse cells | Standard editing rule (Wilson, 1972) |
| Derived flow | Generators break identities such as `smean = sbytes / spkts` | No normal mass in impossible places | Identities hold on >= 95% of real normals |
| Conditional VAE | Need diverse candidates with real category tuples | Categoricals create no false tails | Every learned generator beat the baseline by > 0.10 |
| Over-generate x20 + trim 90% | UNSW normals overlap attacks (`sttl` = 254 in 27%) | Attack-shared modes become tails ECOD flags and regions IForest isolates | +0.092 Kaggle (with sculpt) |
| Sculpt | Trimming over-concentrates mass | Keeps rarer legitimate normals -> fewer false positives | Removing it hurt in combination |
| Range-spreading kappa = 50 | IForest wastes random splits on wide columns with no signal | Near-constant body blinds IForest; thin tail to the real max keeps ECOD calm | +0.036 Kaggle |
| Learned per-column plan | One kappa for all columns; tuning on known types overfits | Shape both detectors accept, gains on seen types only penalised | +0.011 Kaggle |
| Behaviour-based held-out types | Need a stand-in for unseen attack types | Tests whether a change helps types it was not tuned on | Seen-type probes lost up to 0.013 |
| Exact PyOD ECOD rule | Local score must rank files like Kaggle | Optimise for the detector actually used | Spearman rho -0.19 -> +0.83 |
| HistGB + logistic regression critic | A tree critic keys on seen-type thresholds | Two models that err differently must both accept a candidate | +0.010 held-out (local); +0.0035 Kaggle |
| Dedupe | Repeated rows / copies of real rows | No information for IForest, extra ties for ECOD, not synthetic | 9 NSL rows replaced; UNSW none |

## Per-dataset hyperparameters

| Setting | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Generator (`generator`) | conditional VAE | conditional VAE |
| Anomaly model | standard 5-fold HistGB | leave-one-type-out (behaviour types) |
| Trimming critic (`scorer`) | HistGB P(anomaly) + realism (`clf+real`; `clf+lr` gave 0.92482) | HistGB + logistic regression, rank-averaged (`clf+lr`) |
| Default shaping (Pipeline 1) | none | kappa = 50, delta = 0.10, 7 columns |
| `learn_shaping` | off | on, with the plan fixed (`plan=UNSW_PLAN_0_57760`; `P3_RELEARN=1` re-learns) |
| `dedupe` | on | on |
| Seed | 2026 | 2027 |

NSL-KDD normals are already well separated (0.93 with real normals), and every reshaping we tried
lowered its public score, so shaping is off there. UNSW-NB15 normals overlap attacks, so it gets
learned shaping and the second critic. Settings live in `PER_DATASET` (Pipeline 1 part) and
`PER_DATASET_3` (Pipeline 2/3 part) inside `pipeline3.py`.

## Generator choice: CTGAN and GMM were tried, the conditional VAE is best

We compared three learned generators through identical trimming and sculpting: CTGAN, a per-tuple
Gaussian mixture (GMM) and the conditional VAE. All beat the donor-seed baseline on UNSW-NB15 by more
than 0.10, and the **conditional VAE was best** (+0.012 over CTGAN, +0.021 over GMM); on NSL-KDD all
four are within noise. The VAE decodes from the posterior of real normals weighted by (1 - P)^8, so its
candidates stay near safe real neighbourhoods and more of the 20x pool survives trimming. CTGAN samples
from noise and is harder to train on small category tuples; the GMM's Gaussian components smooth over
the spiky count columns that dominate UNSW-NB15. The VAE also needs only NumPy (CTGAN needs PyTorch).

The generator is a per-dataset hyperparameter (`generator` in `PER_DATASET_3`, or `P3_GENERATOR`).
`baseline` is the tuple donor-seed generator without trimming (with dedupe off it reproduces the
Kaggle-scored 0.92940 NSL-KDD baseline file byte for byte); the untrimmed baseline cannot be combined
with `learn_shaping`. Both datasets default to the VAE:

| Generator | NSL-KDD public | UNSW-NB15 public (Pipeline 1 settings) |
|---|---|---|
| **VAE** | **0.92960** | **0.55527** |
| baseline | 0.92940 | 0.43321 |
| GMM | 0.92818 | 0.53440 |
| CTGAN | 0.92473 | 0.54311 |

On NSL-KDD the VAE and the baseline differ by 0.0002, inside the +/-0.005 noise band, so either is defensible;
the baseline also produces about 1.5% internal duplicate rows, which dedupe has to replace.

## Kaggle public scores behind the final settings (UNSW-NB15)

| Milestone | Public AUPRC |
|---|---|
| Tuple donor-seed baseline | 0.43321 |
| VAE + trim + sculpt (Pipeline 0) | 0.52515 |
| + range-spread 7 columns, kappa = 2 | 0.55045 |
| kappa = 50 (same pool) | 0.56135 |
| Pipeline 1 final | 0.56335 |
| Pipeline 2, exact ECOD rule, learned plan | 0.57409 |
| Pipeline 3, HistGB + logistic-regression critic | 0.57760 |
| **Pipeline 3, plan fixed in code, rerun on a second machine** | **0.58821** |

Not adopted (each tested, see `exp_*.py`): regularised HistGB critic (0.57182 UNSW, 0.92937 NSL),
three probes tuned to known attack types (0.55279 / 0.54161 / 0.55386), category reshaping,
decorrelating columns, discretising values (in-sample UNSW 0.582 -> 0.545-0.581), unsupervised critics
(autoencoder, kNN) alone, the HistGB + logistic-regression critic on NSL-KDD (0.92482 vs 0.92935), and
re-learning the UNSW plan on another machine with joint row swapping on (0.56936). Run-to-run noise on the
public board is about +/-0.005, and up to +/-0.01 for UNSW-NB15 reruns on different machines.

## Files

| File | Purpose |
|---|---|
| **`pipeline3.py`** | **Final single-file pipeline** (Pipeline 1 stages + Pipeline 2 learning + Pipeline 3 critic and dedupe) |
| `pipeline1.py` | Over-generate -> trim -> sculpt -> range-spread; also the generator comparison and ablation modes |
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
| `results/` | Metadata of the development runs (NSL-KDD 0.92960, UNSW-NB15 0.57760): settings, learned plan, SHA-256, dedupe counts; `results/pipeline2/` holds the Pipeline 2 runs |

Most experiments take a dataset name and split rotations, e.g. `python exp_critics2.py UNSW-NB15 0 1 2`,
and expect the training CSVs next to the scripts or in the working directory.
