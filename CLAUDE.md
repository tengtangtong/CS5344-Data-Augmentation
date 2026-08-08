# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

CS5344 group project — **Longitudinal Data RUL Prediction Challenge**. Python.

Predict a **binned ordinal Remaining Useful Life class** for each asset from its
static specs plus a variable-length history of operational readouts, scored by a
**cost-sensitive** total cost (lower is better). Two Kaggle competitions:

| Dataset | Classes | Unit | Bins |
|---|---|---|---|
| SCANIA Component X | 5 (`m=4`) | dataset time_step | `>48`, `(24,48]`, `(12,24]`, `(6,12]`, `(0,6]` |
| Backblaze (SMART) | 3 (`m=2`) | days | `≥21`, `[11,20]`, `≤10` |

Higher class = more urgent. Right-censored assets (no failure observed in the
window) are labelled class `0`. See `README.md` for the full formulation, the
cost matrices and the dataset schemas.

## Hard constraints (from the briefing — do not violate)

- **One transferable pipeline for both datasets.** Never propose or build two
  separate pipelines. Dataset-specific *parameters/config* are allowed; forked
  code paths and dataset-specific models are not. Prefer a config-driven design
  where dataset differences (bin thresholds, cost matrix, number of classes,
  column schema) are data, not code.
- **Never touch test data.** Both datasets are public. Do not download external
  copies of the test set, do not use test rows for fitting, feature statistics,
  imputation, scaling, threshold selection, or tuning. Any leakage counts as
  cheating. Fit all transforms on train folds only.
- **Reproducibility.** Seed everything, pin dependencies, and make the submitted
  Kaggle result re-runnable end to end.
- **Optimise expected cost, not accuracy.** Metric is
  `Σ (confusion_matrix ⊙ cost_matrix)`. Under-prediction (late alarm) is much
  more expensive than over-prediction. Class-probability output plus an explicit
  Bayes-risk decision (`argmin_ŷ Σ_y p(y) · C[y, ŷ]`) is the natural fit.
- **Cost robustness.** The hidden test cost matrix may be perturbed **±5–10%**.
  Validate under perturbed cost matrices; do not overfit decision thresholds to
  the exact published table.

## Repo Structure

```
data/raw/         # untouched downloads (gitignored)
data/processed/   # derived features / splits (gitignored)
src/              # shared pipeline code
```

Members previously had personal folders (`Adis/`, `Hans/`, `Raphs/`); work is
now consolidating into `src/`. Team: Adis, Hans, Raphs.

## Conventions

- Data files are gitignored (`csv/tsv/parquet/xlsx/pkl`, all of `data/*`).
  Never `git add -f` a dataset; large/private longitudinal data stays local.
- Keep dataset-specific values (bin thresholds, cost matrices, class counts,
  id/time column names) in config, not scattered through the code.
- Split by **asset** (`vehicle_id` / `serial_number`), never by row — rows from
  the same asset across time must not straddle a train/validation boundary.
- Labels come from a *randomly selected* `t_cur`; only history up to `t_cur` may
  be used as input. No future readouts, no post-`t_cur` aggregates.

## Notes

- `Project_Briefing.pdf` is the source of truth for the task; it is gitignored
  (`*.pdf`). Text extraction needs `pypdf` (`pip install pypdf` into `.venv`) —
  no `poppler`/`pdftotext` on this machine.
