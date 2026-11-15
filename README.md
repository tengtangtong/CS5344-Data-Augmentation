# CS5344 Group 13 - Normal-Only Data Augmentation for Anomaly Detection

We generate **synthetic normal rows** for two network-intrusion datasets, NSL-KDD and UNSW-NB15.
The organisers train two fixed detectors, **ECOD** and **Isolation Forest** (10 seeds x 500 trees),
on our rows only, score hidden validation/test rows, and report

```
S = (AUPRC_ECOD + AUPRC_IForest) / 2
```

The goal is downstream utility: synthetic normal data that helps the fixed detectors rank hidden
attacks above hidden normals, not a faithful copy of the training distribution.

**Team:** Han Teng, Adi, Raph

## Final results

The final submissions come from **Pipeline 3** (`CS5344-Data-Augmentation_final_adi pipeline/adi's pipeline/pipeline3.py`),
one pipeline for both datasets with per-dataset hyperparameters.

| Dataset | Rows submitted | Public AUPRC | Private AUPRC |
|---|---|---|---|
| NSL-KDD | 40,000 | 0.92935 | **0.94973** |
| UNSW-NB15 | 70,000 | 0.58821 | **0.69561** |

Public = hidden validation rows (seen + `A_val` attack types). Private = hidden test rows, which add
further attack types (`A_test`) that no submission could be tuned on.

Pipeline 3 in one line: curate real normals (ENN), fit a conditional VAE, over-generate x20, trim the
90% of candidates that a supervised critic (HistGB, plus logistic regression on UNSW) rates most
attack-like, re-sculpt the density, reshape weak UNSW columns with a learned per-column plan, dedupe and
validate. See the pipeline's own README for the step-by-step reasoning and the ablations.

## Repository layout

```
.
|-- README.md                                   <- this file
|-- .gitignore
|-- data/
|   |-- train_NSL-KDD.csv                       <- released training set (normal + seen attacks)
|   `-- train_UNSW-NB15.csv
|
|-- CS5344-Data-Augmentation_final_adi pipeline/   FINAL SUBMISSION ROUTE (Adi)
|   |-- README.md                               <- project background + how to run Pipeline 3
|   `-- adi's pipeline/
|       |-- pipeline3.py                        <- final single-file pipeline, both datasets
|       |-- pipeline1.py, pipeline2.py          <- earlier versions (stages, learning steps, ablation modes)
|       |-- exp_*.py                            <- experiments: critics, regularisation, decorrelation,
|       |                                          discretisation, category reweighting, two-point shaping
|       |-- check_ecod_rule.py                  <- old vs exact PyOD ECOD rule against Kaggle-scored files
|       |-- p2_robustness.py, gen_overfit.py    <- plan robustness check; known-type probes (negative result)
|       |-- requirements.txt
|       `-- README.md                           <- method, per-dataset settings, generator comparison, scores
|
|-- HT's Code/                                  CONDITIONAL VAE NOTEBOOK ROUTE + ANALYSIS (Han Teng)
|   |-- README.md, CLAUDE.md                    <- task notes and experiment log
|   |-- requirements.txt
|   `-- src/
|       |-- NSL data_preprocessing.ipynb        <- EDA: class balance, categoricals, zero-inflation,
|       |-- UNSW data_preprocessing.ipynb          bounded rates, correlations, PCA, detector diagnostics
|       |-- VAE.ipynb                           <- PyTorch conditional VAE + classifier filter + sculpt/trim
|       |-- pseudo kaggle.ipynb                 <- local stand-in for the leaderboard (ECOD + IForest)
|       |-- failure analysis.ipynb              <- which attacks the detectors miss and why (e.g. sttl = 254)
|       `-- experiments_2026-10-14/             <- candidate dumps, leave-cluster-out proxy, false-alarm study
|
`-- raph_ctgan/                                 CTGAN ROUTE (Raph)
    |-- generate.py                             <- train CTGAN, sample, optional classifier "purify" filter
    |-- data_processing.py                      <- column typing, log transform, zero flags, tail lift, checks
    |-- pipeline3_ctgan.py                      <- Pipeline 3 with Raph's CTGAN improvements switchable
    |-- requirements.txt
    `-- README.md                               <- changes in order, with public scores
```

## The three routes

| Route | Generator | How labels are used | Best public AUPRC (NSL / UNSW) |
|---|---|---|---|
| **Pipeline 3** (final) | Conditional VAE (NumPy) | Critic trims 90% of a x20 candidate pool; learned UNSW column plan | 0.92935 / **0.58821** |
| HT's `VAE.ipynb` | Conditional VAE (PyTorch, gated decoder) | Classifier filter, then sculpt + trim | **0.93758** / 0.535 |
| Raph's CTGAN | CTGAN with log transform, zero flags, tail lift | Classifier "purify" keeps the most normal-looking 80% (UNSW) | 0.92561 / 0.47896 |

All three met the same rule: one code path for both datasets, settings may differ per dataset.
Pipeline 3 was chosen for the final submission because of its UNSW-NB15 score and because every
change in it was checked against held-out attack groups, not only the public board. Inside Pipeline 3
we also compared CTGAN and a Gaussian mixture as the generator; the conditional VAE was best
(UNSW 0.55527 vs CTGAN 0.54311 and GMM 0.53440 under identical trimming).

## Setup

Python 3.8 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Each route has its own `requirements.txt`:

| Route | Install | Main packages |
|---|---|---|
| Pipeline 3 | `pip install -r "CS5344-Data-Augmentation_final_adi pipeline/adi's pipeline/requirements.txt"` | numpy, pandas, scipy, scikit-learn (pyod, ctgan optional) |
| HT's notebooks | `pip install -r "HT's Code/requirements.txt"` | torch, pyod, matplotlib, seaborn, jupyter kernel |
| Raph's CTGAN | `pip install -r raph_ctgan/requirements.txt` | ctgan, torch, pyod |

## How to run

### Final submission: Pipeline 3

`pipeline3.py` reads `train_NSL-KDD.csv` and `train_UNSW-NB15.csv` from the working directory.

```bash
cd "CS5344-Data-Augmentation_final_adi pipeline/adi's pipeline"
cp ../../data/train_NSL-KDD.csv ../../data/train_UNSW-NB15.csv .
python pipeline3.py                         # both datasets, about 10-15 min each on a 2-core CPU
P3_DATASETS=UNSW-NB15 python pipeline3.py   # one dataset
P3_FAST=1 python pipeline3.py               # smoke test, not the submission
```

Output: `final_pipeline3_output_group13/` with `submission_NSL-KDD_p3.csv` (40,000 rows) and
`submission_UNSW-NB15_p3.csv` (70,000 rows). Set `P3_OUT` to change the folder. Other switches
(`P3_GENERATOR`, `P3_SCORER`, `P3_RELEARN`, `P3_JOINT`, `P3_MODE`, ...) are listed in that folder's README.

Reproducibility: runs are seeded and byte-identical on the same machine. Across machines,
floating-point differences change the generated rows, so the learned UNSW column plan is stored in
the code (`UNSW_PLAN_0_57760`) instead of being re-learned. Expect UNSW-NB15 to vary by about
+/-0.01 on a different machine and NSL-KDD by about +/-0.005.

### HT's notebooks

Open the notebooks in `HT's Code/src/` with Jupyter. Before running, edit the config cell:

- `ROOT` is set to the author's local folder (`/Users/han./Desktop/CS5344`); point it at this repo.
- The notebooks expect `data/raw/NSL-KDD_train.csv` and `data/raw/UNSW-NB15_train.csv`; either copy the
  files from `data/` to those names or change the `path` entries.
- In `VAE.ipynb`, set `DATASET = "NSL"` or `"UNSW"` and run all cells; the output goes to
  `data/processed/`. The scripts in `experiments_2026-10-14/` use the same `ROOT`.

### Raph's CTGAN

Run from `raph_ctgan/`:

```bash
cd raph_ctgan
python generate.py --dataset nslkdd --train ../data/train_NSL-KDD.csv --epochs 100 \
    --out outputs/synthetic_nsl-kdd.csv --model-out models/ctgan_nslkdd.pt
python generate.py --dataset unsw --train ../data/train_UNSW-NB15.csv --epochs 100 --purify 0.8 \
    --out outputs/synthetic_unsw-nb15.csv --model-out models/ctgan_unsw.pt
```

See `raph_ctgan/README.md` for the flags and for `pipeline3_ctgan.py`.

## Data rules we followed

- Only the released `train` files are used. Validation, test and Kaggle rows are never read, probed or
  reconstructed.
- Submissions contain only feature rows (plus `id`), no labels, at exactly N rows
  (40,000 NSL-KDD, 70,000 UNSW-NB15). Every value stays inside the real-normal range and every
  category combination occurs in real normals.
- Generated CSVs, trained models and PDFs are not committed (`.gitignore` excludes `*.csv`, `*.pkl`,
  `*.pdf`); the two training files in `data/` are the only CSVs in the repo.

## Datasets

- **NSL-KDD**: one row per network connection; 41 features (38 numeric, 3 categorical:
  `protocol_type`, `service`, `flag`). 30,000 training rows (20,000 normal, 10,000 attacks).
- **UNSW-NB15**: one row per network flow; 42 features (39 numeric, 3 categorical: `proto`,
  `service`, `state`). 50,000 training rows (35,000 normal, 15,000 attacks).

Attack types are split into `A_seen` (training), `A_val` (added in validation) and `A_test` (added in
test), so a method has to help on attack types it has never seen.

