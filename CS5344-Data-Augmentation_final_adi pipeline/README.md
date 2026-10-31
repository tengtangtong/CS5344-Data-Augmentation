# CS5344 Group Project — Tabular Data Augmentation for Anomaly Detection

Designing a **tabular data augmentation/generation method** that produces
normal-only synthetic data to improve downstream anomaly detection, evaluated
on two network-security datasets: NSL-KDD and UNSW-NB15.

## Background

**Anomaly detection** finds data points that deviate from the expected
"normal" pattern — used in fraud detection, industrial monitoring, network
attack detection, and more. Tabular anomalies come in different flavours,
e.g. an **odd value** (a single value far outside its usual range) or a
**rare combination** (individually normal values that co-occur unusually).

**Why augmentation?** Well-designed synthetic normal data can sharpen a
detector's normal/abnormal boundary, fill out a distribution when real
(especially normal) data is scarce, and let sensitive data (e.g. attack logs)
be shared without exposing real records.

## Problem Formulation

### Data

A labeled tabular dataset `(D, y)` with `n` samples, each with `m` feature
columns (numerical or categorical) and a binary label `y_i ∈ {0, 1}`
(`0` = normal, `1` = abnormal). The dataset is split by rows into disjoint
`(D_train, y_train)`, `(D_valid, y_valid)`, `(D_test, y_test)`. Only
`train.csv` is released to participants; `valid.csv`/`test.csv` stay private.

Abnormal rows belong to a specific anomaly type (attack category). The types
are partitioned into three disjoint groups `A_seen`, `A_val`, `A_test`, such
that:

```
types in D_train  =  A_seen
types in D_valid   =  A_seen ∪ A_val
types in D_test    =  A_seen ∪ A_val ∪ A_test
```

So the training set only shows anomaly types participants can observe in
advance; validation adds unseen-during-training types; test further adds
types never seen in training or validation. The anomaly-type label itself is
used only internally by organizers to build/score the splits — it is not
released and not part of `x_i`.

### Task

Using the released `(D_train, y_train)` (both features and labels may be
used), design a generation/augmentation method that produces a fixed-size,
**normal-only** synthetic dataset:

```
D̃ = {x̃_i}, i = 1..N
```

`N` is a fixed submission size. The submission contains only generated
feature vectors — no labels.

### Downstream evaluation

Each submitted `D̃` is used to train a **fixed set of anomaly detectors**
under a training scheme controlled by the organizers:

- **ECOD** — deterministic tail/rank detector
- **IForest** — 10 fixed seeds x 500 trees

Same settings for every team. The trained detectors score the private
validation/test rows (higher score = more abnormal), and performance is
measured with **AUPRC** (Area Under the Precision-Recall Curve):

```
S = (AUPRC_ECOD + AUPRC_IForest) / 2
```

The objective is **downstream utility** — synthetic normal data that helps
the fixed detectors separate normal from abnormal hidden samples — not
faithful reproduction of the training distribution.

## Datasets

### NSL-KDD

Refined version of the KDD Cup 1999 network intrusion dataset. Each row is
one network connection; features describe connection properties (duration,
byte counts, protocol, …). Attacks span many types (e.g. `neptune` SYN-flood
DoS, `ipsweep` probing/surveillance).

- `m = 41` features: 38 numerical + 3 categorical (`protocol_type`,
  `service`, `flag`).
- Feature groups: **basic** connection features, **content** features
  (suspicious-behaviour indicators within a connection), **traffic** features
  (computed over a 2-second window for the same host/service, or over the
  preceding 100 connections for `dst_host_*`).
- Only `train.csv` is released; `valid.csv`/`test.csv` are private.

> Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed
> analysis of the KDD CUP 99 data set.* IEEE CISDA.

### UNSW-NB15

Generated with the IXIA PerfectStorm tool at the Cyber Range Lab of UNSW
Canberra. Each row is one network flow; features describe flow properties
(packet counts, TCP window size, connection state, …). Attacks span several
categories (e.g. `Generic` block-cipher attack, `DoS`).

- `m = 42` features: 39 numerical + 3 categorical (`proto`, `service`,
  `state`; `-` in `service` means not a well-known service).
- Only `train.csv` is released; `valid.csv`/`test.csv` are private.

> Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
> network intrusion detection systems.* MilCIS.

## Evaluation & Leaderboard

- **Public leaderboard** (during the competition): scored on hidden
  *validation* rows, feedback shown immediately, used for development.
- **Private leaderboard** (after the competition): scored on hidden *test*
  rows, hidden during the competition, determines final ranking. Test rows
  may include anomaly types never seen in training or validation — don't
  overfit to the public leaderboard.

Kaggle rank is evidence, not a direct grade conversion. Overall grading
(project component = 60% of module grade: proposal 15%, presentation 20%,
report 25%) also weighs method novelty/transferability, explainability and
robustness, experiment quality, writing/presentation, and code
reproducibility.

## Rules & Constraints

- **One transferable pipeline.** A single generation pipeline must work for
  *both* datasets; parameter settings may differ per dataset, but forked
  code paths or dataset-specific models are not allowed.
- **Normal-only, fixed-size submission.** `D̃` must contain exclusively
  synthetic samples meant to represent the normal class, at exactly `N` rows,
  with no labels included.
- **Deep learning is allowed**, but the method's distributional properties
  and how it produces them must be clearly explainable — this is graded.
- **Reproducibility.** Submit source code after the competition ends so
  results can be reproduced end to end.
- Kaggle allows a **limited number of submissions per day**.

## Team

- Han Teng
- Adi
- Raph

## Repo Structure

```
.
├── data/
│   ├── train_NSL-KDD.csv       # released training sets
│   └── train_UNSW-NB15.csv
├── adi's pipeline/             # final pipeline (Pipeline 3) + experiments
│   ├── pipeline3.py           # ONE pipeline for both datasets → final submissions
│   ├── pipeline1.py           # stages + generator comparison / ablation modes
│   ├── pipeline2.py           # learning steps in modular form (used by experiments)
│   ├── exp_*.py, check_ecod_rule.py, p2_robustness.py, gen_overfit.py   # experiments
│   ├── results/                # metadata of the final runs (settings, learned plans, SHA-256)
│   ├── requirements.txt
│   └── README.md               # method, per-dataset hyperparameters, why each step is there
├── src/
├── CLAUDE.md
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "adi's pipeline/requirements.txt"   # numpy, pandas, scipy, scikit-learn
```

## Usage

```bash
cd "adi's pipeline"
python pipeline3.py          # reads ../data/train_*.csv, writes pipeline3_output/
```

| Output | Rows | Kaggle public AUPRC |
|---|---|---|
| `pipeline3_output/submission_NSL-KDD_p3.csv` | 40,000 | 0.92960 |
| `pipeline3_output/submission_UNSW-NB15_p3.csv` | 70,000 | 0.57760 |

The run is seeded and deterministic (about 10–15 minutes per dataset on a 2-core CPU), and uses only
the released training data. `P3_FAST=1 python pipeline3.py` is a shorter smoke test (about 8 minutes for both datasets). See
`adi's pipeline/README.md` for the method, the per-dataset hyperparameters and the experiments.
