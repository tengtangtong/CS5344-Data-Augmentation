# Pipeline 16 — Over-generate → Trim → Sculpt → Range-spread (+ pooled ensemble)

Normal-only synthetic data for the CS5344 anomaly-detection augmentation task
(NSL-KDD and UNSW-NB15). One transferable pipeline; only per-dataset settings differ.

## Run

```bash
pip install numpy pandas scipy scikit-learn        # required (Python ≥ 3.8, scikit-learn ≥ 1.0)
pip install pyod ctgan                             # optional: exact PyOD ECOD, CTGAN generator
python pipeline16.py                               # MODE = "final" (default)
```

Put `train_NSL-KDD.csv` and `train_UNSW-NB15.csv` next to the script.
Outputs (in `pipeline15_output/`):

| File | Rows |
|---|---|
| `submission_NSL-KDD_final.csv` | 40,000 |
| `submission_UNSW-NB15_final.csv` | 70,000 |
| `<dataset>_final_metadata.json` | settings, SHA-256, derived/spread columns |

Runtime on a laptop CPU: roughly 5–10 min per dataset. Fully seeded and deterministic.

## Pipeline 16 changes (from the Kaggle ablation)

| Kaggle result | Change in pipeline 16 |
|---|---|
| UNSW VAE + spread κ=2: 0.55527 → κ=4: **0.55788** | UNSW default `spread_kappa = 4.0` |
| "Greedy combined" configs (γ=128, pre-filter, sculpt=none, derived off): 0.518–0.547 | not adopted; those factors stay at defaults |
| NSL-KDD: baseline 0.9294, VAE 0.92923–0.92960, GMM 0.92818, CTGAN 0.92473, baseline ablations 0.92905–0.9294 | all within run-to-run noise; NSL-KDD defaults unchanged |
| Same UNSW configuration scored 0.55045 and 0.55527 on two machines | run-to-run noise ≈ ±0.005 → new **pooled ensemble** option to reduce it |

**Pooled ensemble** (`pool_members`): several generators or seeds each contribute an equal share of
every category tuple's candidates *before* trimming, so trimming and sculpting choose from a more
diverse pool. Members are written `"<gen>"` or `"<gen>#k"` (seed offset k), e.g.
`pool_members=["vae", "vae#1", "vae#2"]`. Locally (UNSW, 2 splits): VAE seed pool +0.001 / +0.001;
VAE + GMM pool −0.005 / −0.006 (not recommended).

Full Kaggle generator comparison (MODE="generate", pipeline 16 defaults):

| | NSL-KDD | UNSW-NB15 |
|---|---|---|
| baseline tuple donor-seed | 0.9294 | 0.43321 |
| VAE | 0.92923 | 0.55527 |
| GMM | 0.92818 | 0.5344 |
| CTGAN | 0.92473 | 0.54311 |

### Detector-aware options added after the κ sweep (all off by default)

| Option | What it does | Local evidence (UNSW, held-out test) |
|---|---|---|
| `anchor_frac=0.005, anchor_keep=8` | 0.5% of rows carry the real-normal maximum on every column except the 8 most informative; IForest's uniform thresholds on those columns then mostly peel off just these rows, so it effectively ignores weak columns while ECOD's ECDFs barely move | +0.003 / +0.007 overall; seen and unseen both up |
| `tilt_plan={col: κ}` | per-column quantile tilt v = Q_real(u^κ), learned by `learn_tilt_plan()` (coordinate ascent on ECOD+IForest AUPRC over a seen-type validation split) | seen types +0.029, unseen −0.05, overall −0.008 → public-board oriented, private-board risk |
| greedy keep/spread/const per column | (prototype, not kept) | overall −0.004 |

Kaggle κ sweep (same candidate pool): κ=4 0.55102 → κ=6 0.55427.

## Final submissions (Kaggle-validated)

| Dataset | File (`MODE="final"`) | Configuration | Public score |
|---|---|---|---|
| NSL-KDD | `submission_NSL-KDD_final.csv` | VAE, trim 90% (classifier + realism), sculpt, no spreading | **0.92960** |
| UNSW-NB15 | `submission_UNSW-NB15_final.csv` | VAE, leave-one-type-out anomaly model, trim 90% (classifier), sculpt, range-spread 7 count columns with κ = 50 | **0.56135** |

UNSW κ ladder (same candidate pool, 7 columns, δ = 0.10):

| κ | Share of spread values at column minimum | Public score |
|---|---|---|
| 4 | 11% | 0.55102 |
| 6 | – | 0.55427 |
| **50** | 87% | **0.56135** |
| 100 | 92% | 0.56049 |
| 1000 | 99% | 0.56103 |

Column count at κ = 50: 5 columns 0.55422, **7 columns 0.56135**, 15 columns 0.53118.
NSL-KDD spreading of the 6 count columns at κ = 20: 0.92932 (no gain over 0.92960).

## Stages

1. **Curation** — drop real normals whose 10 nearest labelled neighbours are mostly anomalies (ENN).
2. **Derived-feature flow** — audit row-local identities on real normals (≥95% agreement);
   verified derived columns (UNSW: `tcprtt`, `smean`, `dmean`) and constant columns
   (NSL-KDD: `wrong_fragment`, `num_outbound_cmds`, `is_host_login`) are removed before
   generation and restored afterwards.
3. **Anomaly model** — HistGradientBoosting on real normal vs `is_anomaly`; out-of-fold
   P(anomaly) for real normals (target weights) and P(anomaly) for every candidate (trimming).
   UNSW uses a leave-one-type-out ensemble (5 pseudo attack types).
4. **Generator** — mixed-type conditional VAE (NumPy): conditioned on the categorical tuple;
   continuous columns quantile-normalised with a Gaussian head, integer columns with ≤10 levels
   as softmax heads; samples decoded from the posterior of weighted real normals.
5. **Over-generate ×20, trim 90%** — keep the 10% of candidates per categorical tuple with the
   lowest trimming score (NSL-KDD: rank of P(anomaly) + realism discriminator; UNSW: P(anomaly)).
6. **Sculpt** — per-tuple quotas from anomaly-aware weights (1 − P)^γ, then
   sampling-importance-resampling with a target-vs-pool density ratio.
7. **Range-spread (UNSW only)** — weak (|AUC − 0.5| < 0.10), heavy-tailed (skew > 3), non-fraction
   columns are mapped rank-preservingly onto the real-normal range: v = lo + u²(hi − lo).
   UNSW columns: `spkts, dpkts, sbytes, sloss, dloss, response_body_len, ct_src_dport_ltm`.
8. **Validate** — exact schema, row count, no NaN, no invented categories or category tuples,
   values inside the real-normal range, no exact copies of real rows.

## Kaggle public scores behind the final settings

| Dataset | Configuration | Public score |
|---|---|---|
| NSL-KDD | baseline (tuple donor-seed) | 0.92954 |
| NSL-KDD | GMM | 0.92882 |
| NSL-KDD | **VAE (final)** | **0.92960** |
| NSL-KDD | VAE + spread (10 cols, incl. error-rate fractions) | 0.92192 |
| UNSW-NB15 | baseline | 0.43329 |
| UNSW-NB15 | GMM | 0.51694 |
| UNSW-NB15 | VAE | 0.52515 |
| UNSW-NB15 | VAE + spread δ=0.20 (20 cols) | 0.50276 |
| UNSW-NB15 | VAE + spread δ=0.15 (15 cols) | 0.54466 |
| UNSW-NB15 | **VAE + spread δ=0.10 (7 cols) (final)** | **0.55045** |

## Trim-placement ablation (`MODE = "trim_ablation"`)

The anomaly classifier is always trained on the released labelled data (real normals vs
`is_anomaly`). This mode varies **where** its P(anomaly) removes rows:

| Variant | Pre-trim (real input normals) | Post-trim (×20 over-generated candidates) |
|---|---|---|
| `A_post_only` (final) | – | keep best 10% |
| `B_none` | – | – (generate N directly) |
| `C_pre10_only` | drop top 10% by P(anomaly) | – |
| `D_pre25_only` | drop top 25% | – |
| `E_pre10_post` | drop top 10% | keep best 10% |
| `F_pre25_post` | drop top 25% | keep best 10% |

Generators: `TRIM_GENERATORS = ["vae", "gmm", "ctgan"]` (CTGAN runs only if `ctgan` is installed;
`TRIM_CTGAN_EPOCHS` shortens it). Writes `submission_<dataset>_<gen>_trim-<variant>.csv` and
`<dataset>_trim_ablation.csv` (local proxy score per variant; fill `KAGGLE_TRIM_SCORES` with public
scores to keep them alongside).

Local proxy (1 split, full-size pipeline):

| | NSL-KDD VAE | NSL-KDD GMM | UNSW VAE | UNSW GMM |
|---|---|---|---|---|
| A post only | 0.9357 | 0.9364 | **0.5610** | 0.5533 |
| B none | 0.9341 | 0.9332 | 0.4772 | 0.4772 |
| C pre 10% | 0.9322 | 0.9301 | 0.4843 | 0.4919 |
| D pre 25% | 0.9333 | 0.9310 | 0.5281 | 0.5259 |
| E pre 10% + post | 0.9346 | 0.9308 | 0.5562 | **0.5535** |
| F pre 25% + post | 0.9346 | 0.9320 | 0.5432 | 0.5352 |

Only 0.34% of NSL-KDD normals have P(anomaly) > 0.01, so pre-trimming there removes ordinary
normals; on UNSW about a fifth of normals look attack-like, so pre-trimming removes real overlap.

## Other modes

- `MODE = "generate"` — writes baseline / VAE / CTGAN / GMM candidates per dataset plus local
  proxy scores (CTGAN needs `pip install ctgan`).
- `MODE = "ablate"` — reads `KAGGLE_SCORES`, picks the best generator, runs a one-factor-at-a-time
  ablation on a local proxy (3 splits, pseudo-unseen anomaly types held out) and exports the top
  configurations.

## Known failure modes (for the report)

- Range-spreading helps only on columns that carry no attack signal; spreading bounded
  error-rate fractions (NSL-KDD) or timing columns (UNSW δ=0.20) lowered the public score.
- The local proxy ranked generators correctly (Spearman 1.0 on UNSW) but mis-ranked the
  spreading variants; Kaggle feedback was decisive there.
- ECOD and IForest want opposite things from weak columns (IForest benefits from constant
  columns, ECOD penalises them), which bounds what data shaping alone can achieve.
