# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

CS5344 group project — **Tabular Data Augmentation for Anomaly Detection**.
Python. Team: Han Teng, Adi, Raph.

Given a labeled training set (normal + abnormal rows), design a data
augmentation/generation method that produces a **fixed-size (N), normal-only**
synthetic dataset. The synthetic data is submitted to Kaggle, where it is used
to train a **fixed set of downstream anomaly detectors** (ECOD, and IForest —
10 fixed seeds x 500 trees) under a training scheme the organizers control.
Those detectors then score private validation/test rows, and performance is
measured by **AUPRC** (Area Under the Precision-Recall Curve):

```
S = (AUPRC_ECOD + AUPRC_IForest) / 2
```

The goal is **downstream utility**, not distributional fidelity — synthetic
normal data that sharpens the detectors' normal/abnormal boundary scores
higher than synthetic data that merely reproduces the training distribution.

Two Kaggle competitions, each an independent instantiation of the same task:

| Dataset | Rows are | Features (m) | Categorical cols |
|---|---|---|---|
| NSL-KDD | one network connection | 41 (38 numerical + 3 categorical) | `protocol_type`, `service`, `flag` |
| UNSW-NB15 | one network flow | 42 (39 numerical + 3 categorical) | `proto`, `service`, `state` |

Only `train.csv` is released per dataset; `valid.csv`/`test.csv` stay private
on the organizers' side. Abnormal rows belong to disjoint anomaly-type groups
`A_seen` / `A_val` / `A_test`: the training set only shows `A_seen` types, the
private validation set adds `A_val` (unseen-during-training) types, and the
private test set further adds `A_test` (unseen-during-either) types. The
anomaly-type label itself (attack cat/class) is used only by organizers to
build/score the splits — it is not released and not part of `x_i`. Submitted
data must not include anomaly labels, only generated feature rows.

See `CS5344_Project_Briefing_2627Sem1.pdf` and
`CS5344_anomaly_detection_problem_formulation.pdf` for the full formulation
(both gitignored, `*.pdf`).

## Hard constraints (from the briefing — do not violate)

- **One transferable pipeline for both datasets.** Never propose or build two
  separate pipelines. Dataset-specific *parameters/config* are allowed; forked
  code paths and dataset-specific models are not. Prefer a config-driven design
  where dataset differences (feature schema, categorical columns, m) are data,
  not code.
- **Normal-only synthetic output.** The submitted synthetic dataset `D̃` must
  consist exclusively of samples representing the *normal* class, at exactly
  the fixed submission size `N`. Never include abnormal synthetic rows or
  labels in the submission.
- **Never touch validation/test data.** `valid.csv`/`test.csv` are private and
  not released — there is nothing to leak from them, but do not attempt to
  probe, reconstruct, or fit against hidden Kaggle rows either. Fit the
  generation method only on the released `train.csv` for each dataset.
- **Optimize downstream detector utility, not train-set resemblance.** A
  generator that reproduces the training distribution exactly is not
  necessarily the objective — the score is AUPRC of *fixed, externally defined*
  detectors (ECOD, IForest) trained on your synthetic data. Design and validate
  with that downstream objective in mind, not just marginal/joint fit metrics.
- **Explainability requirement.** If using deep-learning generation methods,
  be prepared to clearly explain the distributional properties of the
  generated data and how the method produces them — this is graded.
- **Reproducibility.** Seed everything, pin dependencies, and submit source
  code after the competition ends so results are re-runnable end to end.

## Repo Structure

```
data/raw/         # untouched downloads (gitignored)
data/processed/   # derived features / splits (gitignored)
src/              # shared pipeline code
```

## Conventions

- Data files are gitignored (`csv/tsv/parquet/xlsx/pkl`, all of `data/*`).
  Never `git add -f` a dataset; raw/processed data stays local.
- Keep dataset-specific values (categorical column names, m, file paths) in
  config, not scattered through the code.
- Categorical columns (`protocol_type`/`service`/`flag` for NSL-KDD,
  `proto`/`service`/`state` for UNSW-NB15) need explicit handling in any
  generative method — don't assume all columns are numerical.

## Grading (for context, not something to optimize code around)

Project component is 60% of the module grade: proposal writeup (15%), final
presentation (20%), final report (25%). Kaggle leaderboard rank is evidence,
not a direct grade conversion — graders also weigh method novelty and
transferability, explainability/robustness, experiment quality, writing, and
code reproducibility. Don't overfit to the public leaderboard; hidden test
rows may include anomaly types never seen in training or validation.

## Notes

- PDFs (`CS5344_Project_Briefing_2627Sem1.pdf`,
  `CS5344_anomaly_detection_problem_formulation.pdf`) are gitignored (`*.pdf`)
  and are the source of truth for the task. Text extraction needs `pypdf`
  (`pip install pypdf` into `.venv`) — no `poppler`/`pdftotext` on this
  machine.
- Starter kit was not yet available as of the briefing date; check for it
  before scaffolding submission-format code from scratch.
