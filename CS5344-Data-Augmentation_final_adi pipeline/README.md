# Final submission route: Pipeline 3 (Adi)

This folder holds the code behind Group 13's final Kaggle submissions. The project background, the
other two routes and the repo layout are in the [root README](../README.md).

| Dataset | Rows | Public AUPRC | Private AUPRC |
|---|---|---|---|
| NSL-KDD | 40,000 | 0.92935 | **0.94973** |
| UNSW-NB15 | 70,000 | 0.58821 | **0.69561** |

## Contents

```
CS5344-Data-Augmentation_final_adi pipeline/
|-- README.md                  <- this file
`-- adi's pipeline/
    |-- pipeline3.py           <- ONE pipeline for both datasets -> final submissions
    |-- pipeline1.py           <- stages, generator comparison and ablation modes
    |-- pipeline2.py           <- learning steps in modular form (used by the experiments)
    |-- exp_*.py               <- experiments (critics, regularisation, decorrelation,
    |                             discretisation, category reweighting, two-point shaping)
    |-- check_ecod_rule.py     <- old vs exact PyOD ECOD rule against Kaggle-scored files
    |-- p2_robustness.py       <- scores a learned plan on further held-out splits
    |-- gen_overfit.py         <- three probes tuned to known attack types (negative result)
    |-- requirements.txt
    `-- README.md              <- method, why each step is there, per-dataset settings, all scores
```

## Run it

```bash
pip install -r "adi's pipeline/requirements.txt"     # numpy, pandas, scipy, scikit-learn
cd "adi's pipeline"
cp ../../data/train_NSL-KDD.csv ../../data/train_UNSW-NB15.csv .
python pipeline3.py
```

Output: `final_pipeline3_output_group13/submission_NSL-KDD_p3.csv` and
`final_pipeline3_output_group13/submission_UNSW-NB15_p3.csv` (set `P3_OUT` to change the folder).
A full run takes about 10-15 minutes per dataset on a 2-core CPU; `P3_FAST=1 python pipeline3.py` is a
shorter smoke test (not the submission).

Runs are seeded and byte-identical on the same machine. On another machine, floating-point
differences change the generated rows, so the learned UNSW-NB15 column plan is stored in the code
(`UNSW_PLAN_0_57760`) rather than re-learned. Expect UNSW-NB15 to move by about +/-0.01 and NSL-KDD by
about +/-0.005 across machines.

Only the released training files are used; validation, test and Kaggle rows are never read.
See [`adi's pipeline/README.md`](adi's%20pipeline/README.md) for the full method and experiments.
