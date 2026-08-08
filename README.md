# CS5344 Group Project — Longitudinal Data RUL Prediction Challenge

Predicting **binned Remaining Useful Life (RUL)** from longitudinal (time-indexed
tabular) asset data, evaluated with a **cost-sensitive** metric on two datasets:
SCANIA Component X and Backblaze hard drives.

## Background

**Longitudinal data** is tabular data with a time dimension: the same sample is
tracked at multiple points in time, so each record combines a rich feature set
with temporal progression.

**Remaining Useful Life (RUL)** is how much time is left before an asset fails or
must be replaced. Predicting failures early reduces unplanned downtime, avoids
emergency repairs, and allows maintenance to be scheduled efficiently.

## Problem Formulation

### Assets and longitudinal readouts

We observe a fleet of assets. Each asset `i` has:

- **static specifications** `s_i` (optional metadata), and
- a **variable-length sequence of operational readouts**
  `R_i = {(t_i,1, x_i,1), …, (t_i,Li, x_i,Li)}`, where `Li` varies per asset,
  `t_i,k` is a dataset-specific timestamp and `x_i,k` is the feature vector
  measured at that time.

### Event time and RUL

Let `T_i` be the failure/event time of asset `i` (component replacement time,
drive failure date). Remaining useful life at time `t` is

```
RUL_i(t) = max(T_i − t, 0)
```

In real logs many assets do not fail within the observation window; those cases
are **right-censored** (`T_i` unknown).

### Binned ordinal labels

Instead of regressing continuous RUL, it is discretised into `m + 1` ordered
bins using thresholds `0 < τ0 < … < τm`. At a *randomly selected* current time
`t_i^cur` the label `y ∈ {0, …, m}` is

```
y_i = m      if 0 < RUL_i(t_i^cur) ≤ τ0
      m − 1  if τ0 < RUL_i(t_i^cur) ≤ τ1
      …
      0      if RUL_i(t_i^cur) > τm
```

Higher `y` means more urgent. Right-censored assets get `y = 0`, interpreted as
"not expected to fail within the short-term horizon `τm`" given the available
evidence.

### Objective

Given an asset's static specifications `s_i` and its full history `R_i` up to the
randomly selected time `t_i^cur`, predict the binned RUL class:

```
f(s_i, R_i) → ŷ_i ∈ {0, …, m}
```

## Datasets

### SCANIA Component X

5 classes (`m = 4`), time unit as given in the dataset:

| Class | RUL range | |
|---|---|---|
| 0 | > 48 | safe |
| 1 | (24, 48] | warning |
| 2 | (12, 24] | warning |
| 3 | (6, 12] | warning |
| 4 | (0, 6] | urgent |

Two tables:

- **Specifications** — `vehicle_id`, `Spec_0 … Spec_7` (categorical, e.g. `Cat0`,
  `Cat1`, …).
- **Operational readouts** — `vehicle_id`, `time_step`, plus many numeric sensor
  columns (`171_0`, `666_0`, `427_0`, `837_0`, `167_0 … 167_9`, `309_0`, …).

> Lindgren, T., Steinert, O., Andersson Reyna, O., Kharazian, Z., & Magnússon, S.
> (2024). *SCANIA Component X Dataset: A Real-World Multivariate Time Series
> Dataset for Predictive Maintenance* (Version 2) [Data set]. Scania CV AB.
> https://doi.org/10.5878/jvb5-d390

### Backblaze

A large-scale hard-drive reliability dataset with daily SMART statistics.
3 classes (`m = 2`), unit = days:

| Class | RUL range | |
|---|---|---|
| 0 | ≥ 21 | safe |
| 1 | [11, 20] | warning |
| 2 | ≤ 10 | urgent |

Columns: `date`, `serial_number`, `model`, `capacity_bytes`, `failure`, and
`smart_*_normalized` / `smart_*_raw` pairs.

- **Train set** — records of failed drives from the start date to the failure date.
- **Test / validation set** — records of drives from the start date to the
  prediction date.

> https://www.backblaze.com/b2/hard-drive-test-data.html

## Evaluation

Scoring is **cost-sensitive**:

```
Total_cost = Σ (confusion_matrix ⊙ cost_matrix)      # lower is better
```

Mistakes are not equal — a **missed / late alarm** costs far more than an
**early alarm / unnecessary check**.

**SCANIA** (rows = actual, cols = predicted):

|  | 0 `>48` | 1 `(24,48]` | 2 `(12,24]` | 3 `(6,12]` | 4 `(0,6]` |
|---|---|---|---|---|---|
| **0** `>48` | 0 | 20 | 30 | 40 | 50 |
| **1** `(24,48]` | 200 | 0 | 20 | 30 | 40 |
| **2** `(12,24]` | 300 | 200 | 0 | 20 | 30 |
| **3** `(6,12]` | 400 | 300 | 200 | 0 | 20 |
| **4** `(0,6]` | 500 | 400 | 300 | 200 | 0 |

**Backblaze**:

|  | 0 `≥21` | 1 `[11,20]` | 2 `≤10` |
|---|---|---|---|
| **0** `≥21` | 0 | 1 | 3 |
| **1** `[11,20]` | 4 | 0 | 2 |
| **2** `≤10` | 15 | 5 | 0 |

### Cost fluctuation and robustness

In real predictive maintenance costs are not fixed — labour, downtime penalties,
spare parts and logistics all change, so the "true" cost matrix is uncertain and
approximate. The hidden test set **may perturb the cost matrix by ±5–10%** to
simulate this. The goal is a method that is robust, not one tuned to a single
exact cost table.

## Rules & Constraints

- **One transferable pipeline.** A single model pipeline must work for *both*
  datasets. Two completely different pipelines are not allowed; different
  *parameter settings* per dataset within the same pipeline are fine.
- **No test data.** Both datasets are public, so similar test data may exist
  online. Do **not** download test data, and never use it for training or
  tuning — this is treated as cheating.
- **Reproducibility.** The pipeline must reproduce the results submitted to
  Kaggle.
- Kaggle allows a **limited number of submissions per day**.

## Team

- Adis
- Hans
- Raphs

## Repo Structure

```
.
├── data/
│   ├── raw/            # untouched downloads (gitignored)
│   └── processed/      # derived features / splits (gitignored)
├── src/                # shared pipeline code
├── Project_Briefing.pdf
├── CLAUDE.md
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # TODO: create
```

## Usage

<!-- TODO: fill in once the pipeline entry points exist -->
