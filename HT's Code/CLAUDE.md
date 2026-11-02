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
data/processed/   # synthetic CSVs / submissions (gitignored)
src/              # notebooks: EDA x2, VAE.ipynb, pseudo kaggle.ipynb, failure analysis.ipynb
src/experiments_2026-10-14/  # scratch scripts: leave-cluster-out eval, false-alarm anatomy
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
- `torch` is installed in `.venv` (for the VAE); versions pinned in
  `requirements.txt`.

## ▶ Where we left off (2026-10-14, end of day) — read this first

### Current best per dataset

| Dataset | Best Kaggle | File (`data/processed/`) | Recipe | Top team |
|---|---|---|---|---|
| NSL | **0.93758 (1st place, public LB)** | `NSL_TRY14_temp8.0.csv` | `VAE.ipynb` NSL defaults: clf 0.5, d0, **temperature 8.0**, no shaping | us |
| UNSW | **0.535** (seed 43 = 0.520, ≈0.527 expected) | `TRY7_UNSW_trim75.csv` | `VAE.ipynb` UNSW defaults: clf 0.01, d0, sculpt 0.35, trim 0.75 | ~0.70 |

Both are produced by running `VAE.ipynb` unchanged (only `DATASET` set); both verified byte for byte.

### Start here next session (2026-10-15)

- **UNSW score work is effectively closed.** On 2026-10-14 four more ideas lost to TRY7: per-column trim
  (Kaggle), `sttl=254` rescue (local), ECOD+IForest trim (Kaggle), and earlier the connection-type mix.
  The user still wants to keep trying UNSW, so if a new idea comes up:
  1. Screen it with the leave-cluster-out local test first (scripts in `src/experiments_2026-10-14/`,
     see below). Drop it if it's clearly worse there.
  2. Then spend exactly 2 Kaggle submissions (seeds 42 + 43). Win = beat 0.535 / 0.520 on the same seed.
- **Ideas not yet tried** (all low odds, none tested):
  - Re-tune `sculpt_keep` jointly with trim 0.75 (sculpt 0.35 was picked before trim 0.75 existed).
    Attack-aware, so expect a shrunken Kaggle gain.
  - Loosen only `dur` / `sjit` (they flag false alarms 94% / 80% vs caught attacks 57% / 56% for ECOD).
    Caution: per-column trim was the same family and failed.
- **Otherwise move to the report:** the analysis below (why tight wins on UNSW, why wide wins on NSL, why
  selective tweaks fail, why the old pseudo kaggle misled us) is ready to turn into the
  analysis / negative-results section. The copula baseline is still open.
- `src/experiments_2026-10-14/` holds the scratch scripts behind today's results (`lib.py` shared setup,
  `dump_cands.py` / `dump_temp.py` dump scored raw VAE candidates, `loco.py` leave-cluster-out evaluator,
  `fa.py` false-alarm anatomy). They expect the candidate pickles (about 600 MB each, not kept) in the
  working dir: rerun `dump_cands.py <seed> cands_s<seed>.pkl` from that folder first.

### Trim by ECOD + IForest: no win on Kaggle (2026-10-14), key removed

- Tried `trim_by="ecod+iforest"` in stage 6b: rows ranked by ECOD and by IForest (500 trees, fit on real
  normals), with the two ranks added, so the trim also drops unusual column combinations. Label-free.
- TRY22 (s42) = 0.52455 vs TRY7 0.535 (−0.010); TRY23 (s43) = 0.52227 vs TRY11 0.52037 (+0.002).
  Mean 0.523 vs 0.527: no gain, within noise. The key was removed (`VAE.ipynb` is back to the pre-edit version).
- **UNSW score work is closed.** Every known lever has been tried. Final picks stay TRY7 + TRY11.

### Leave-cluster-out pseudo kaggle: a better local test, still not good enough (2026-10-14)

- **Idea:** mimic Kaggle's unseen attack types locally. KMeans(6) on UNSW attacks (log1p numerics +
  one-hot cats, `random_state=0`) gives 4 big groups: (0) tcp/-/FIN + mail 8.0k, (1) http/FIN 3.3k,
  (2) udp/-/INT 1.9k, (3) dns 1.5k. Each round hides one group entirely: the method (filter clf +
  sculpt) sees half A of the other groups, and eval = all normals + half B of the other groups + the
  hidden group. Score = mean S over the 4 rounds.
- **Calibration:** 16 past UNSW configs rebuilt from saved VAE candidates (20k-row synthetic sets,
  3 IForest seeds) and compared with their Kaggle scores. Rank agreement (Spearman):

  | Local test | All 16 configs | 14 shaped configs only |
  |---|---|---|
  | old holdout pseudo kaggle | +0.30 | **−0.04** (no better than random) |
  | leave-cluster-out (mean of 4 rounds) | **+0.57** | +0.36 |

  Pearson: 0.73 vs 0.87. IForest drives the agreement (rank corr. 0.57); ECOD doesn't (0.10).
- **Takeaways:** (1) the old pseudo kaggle could not tell the shaped configs apart at all, which
  explains every past local-vs-Kaggle surprise. (2) Leave-cluster-out is better and catches big losers
  (clf 0.5, clf 0.01, temperature 2.0), but it still misses the trim 0.75 peak (trim sweep is flat
  0.444 -> 0.437 locally vs Kaggle peak at 0.75). Kaggle noise (±0.015) caps how well any local test
  can rank the top configs. Use it to screen out clear losers, not to pick winners.
- Script: `src/experiments_2026-10-14/loco.py`. Good report material on validation.

### False-alarm anatomy of TRY7 (2026-10-14, guards trained on TRY7, scored on all train rows)

- Locally **ECOD is the weak guard**: AUPRC 0.50 (half of its top-15k are real normals) vs IForest 0.685
  (36%). IForest also drives the local-vs-Kaggle agreement.
- `sttl=254` normals are 55% (ECOD) / 69% (IForest) of the false alarms. The rest:
  - ECOD (3.3k): `tcp/http/FIN` 26%, `tcp/-/CON` 25%, `tcp/-/FIN` 22%, `tcp/ftp/FIN` 12%, `arp/*` 9%.
    Types missing from the synthetic data get flagged wholesale (arp, ospf 100%; tcp/-/CON 69%; ftp/FIN 77%),
    because an unseen one-hot category lands far past the synthetic range in that column.
    Out-of-range columns: `dur` (94% of false alarms vs 57% of caught attacks) and `sjit` (80% vs 56%)
    flag normals more than attacks. Half of *all* real normals are out of range on the timing columns
    (`sinpkt`, `rate`, `dinpkt`): TRY7 is extremely tight.
  - IForest (1.7k): `tcp/http/FIN` 73%, `tcp/-/CON` 25%. All in the timing/RTT columns.
- **Hypothesis #3 is false:** IForest does *not* flag rare clean types just for being rare. `udp/dns/CON`,
  `tcp/-/REQ`, `arp` are 0% flagged by IForest despite 0-2% synthetic share. Only ECOD punishes missing
  categories.
- **Why restoring clean types failed on Kaggle (TRY13-16), probable explanation:** "missing category -> ECOD
  alarm" also fires on attack-only protos/services (`unas`, `sctp`, ...), which unseen attack types likely use.
  A narrow category mix is good for unseen types even though it costs false alarms on rare normal types.
  "Clean" is only measured against seen attacks.

### `data/processed/` was accidentally deleted on 2026-10-14 and partly regenerated

Rebuilt with `VAE.ipynb` (deterministic): `TRY7_UNSW_trim75.csv`, `TRY11_UNSW_trim75_seed43.csv`,
`NSL_TRY14_temp8.0.csv`, `NSL_TRY12_temp5.0.csv` (temperature 5.0), and
`synthetic_UNSW_clf0.01_holdout.csv` (clf 0.01, no sculpt/trim, `clf_holdout=True`). The pre-edit notebook
reproduced TRY7 byte for byte on 2026-10-13, and the current one reproduces the pre-edit notebook, so
TRY7/TRY11/NSL TRY14 are the submitted files. The holdout file was originally made by an older version of
the notebook, so it may differ slightly. All other old submission files are gone; the table below is the record.

### Per-column trim, `sep` rule: FAILED on Kaggle (2026-10-14)

TRY18 = 0.52405 (vs TRY7 0.535, −0.011), TRY19 = 0.5033 (vs TRY11 0.520, −0.017). Worse on both
seeds. So keeping the spread of low-separation columns (`sbytes`, `dbytes`, ...) hurts: tightening
*every* column is part of why trim works, even columns that flag more normals than attacks locally.
`audit` (TRY20/21) untested; it is a milder version of the same idea. Details of the idea below.

#### The idea (script `src/percol_trim.py` and TRY18–21 files deleted 2026-10-14)

**Idea:** the label-free trim tightens *every* column, but some columns barely separate attacks from
normals. Tightening those only makes real normals look out of range (false alarms). So the trim's
ECOD is fitted on the "catcher" columns only, and the other numeric columns keep the spread they
have after sculpting. Categoricals always stay in the trim.

**Two rules** for choosing which numeric columns trim ignores:
- `sep`: ignore columns where |2·AUC − 1| < 0.2 (normals vs train attacks). Decided up front from
  the data, one threshold. This drops 19 columns, incl. `sbytes`, `dbytes`, `spkts`, `dloss`.
- `audit`: two passes. Do the usual trim, then ignore columns whose output flags more real normals
  than attacks. That's 5–6 columns (`spkts`, `sbytes`, `dbytes`, `sloss`, `dloss`).

**What it does to the data:** real normals / attacks flagged out of range per column, seed 42:

| Column | TRY7 | `sep` |
|---|---|---|
| `dbytes` | 35% / 28% | 16% / 26% |
| `sbytes` | 29% / 25% | 16% / 23% |
| `spkts` | 14% / 7% | 5% / 3% |
| `sttl`, `tcprtt`, `ct_srv_dst` | unchanged | unchanged |

**Files** (all 70k rows, format checked):

| File | Rule | Seed |
|---|---|---|
| `TRY18_UNSW_percol_sep_s42.csv` | sep 0.2 | 42 |
| `TRY19_UNSW_percol_sep_s43.csv` | sep 0.2 | 43 |
| `TRY20_UNSW_percol_audit_s42.csv` | audit | 42 |
| `TRY21_UNSW_percol_audit_s43.csv` | audit | 43 |

- **Submit TRY18 + TRY19 first.** The `sep` rule is cleaner for the report, since it's one
  rule decided up front.
- A rule wins only if **both seeds beat TRY7/TRY11 on the same seed** (0.535 / 0.520).
- If `sep` wins:
  - Move the script's `trim()` into `VAE.ipynb` stage 6b as a config key (e.g.
    `trim_sep=0.2`, 0 = all columns).
  - Check that the UNSW defaults then reproduce TRY18.
  - Delete `percol_trim.py`.
- If neither rule wins: delete the script and record it as a negative result.
- `python src/percol_trim.py --dataset UNSW --rule {none,sep,audit} --seed 42 --out X.csv`
  - It runs `VAE.ipynb`'s own code cells, so the generator is identical.
  - `--rule none` reproduces `TRY7_UNSW_trim75.csv` byte for byte (verified).
  - One run takes a few minutes; 5 in parallel is fine.

### Final picks (2 per competition, for the hidden-test ranking)
- **NSL:** TRY14 (temperature 8.0) + TRY12 (5.0). Both are on the plateau, and 5.0 is the less
  extreme backup.
- **UNSW:** TRY7 + TRY11 (same config, two seeds). Every 2026-10-14 challenger lost.
- Don't overfit the public LB: the hidden test has unseen attack types.

### Still open
- **UNSW `sttl=254` rescue: tested locally 2026-10-14, FAILED, not submitted, no code kept.**
  - Finding: the clf 0.01 filter is *not* what removes 254 rows (2.9% of filter survivors are high-TTL);
    **sculpting removes all of them** (2.9% -> 0.0%). So a fix needs a reserved share.
  - Tried: a "rescue lane" of generated high-TTL rows (`sttl>=200`, the d0 VAE smears 254 into 250-255)
    with a looser classifier threshold (0.2 / 0.3, clf on half A), shaped within the lane (sculpt+trim
    or trim only), mixed in at 5% / 10% / 20% of N. Pseudo kaggle holdout, 2 seeds, 5 IForest seeds.
  - Baseline (TRY7 recipe) S 0.432 / 0.440. **Every variant lower** (best rt 0.3 sculpt+trim 5-10%:
    0.427-0.431 / 0.414-0.425; trim-only lanes 0.355-0.397). ECOD flat, IForest drops 0.50 -> 0.40-0.47.
  - Why: the guards judge one column at a time. Any 254 rows in the data teach them "sttl=254 is
    normal", which also hides the 72% of attacks that have sttl=254. The 0.62 ceiling needs a
    *combination* test (254 AND other columns normal), which ECOD/IForest can't do. Pseudo kaggle
    favours attack-aware ideas, so failing even there is strong evidence. Treat the ceiling as
    unreachable with normal-only data for these detectors.
- **Report story** (one pipeline, a per-column "tightness" knob):
  - NSL: the VAE is too narrow at temperature 1 (std 0.82× real), so widen it.
  - UNSW: normals overlap attacks, so be tight, in *every* column (selective tightening, per-column trim and
    restoring clean types all lost). A narrow category mix also lets ECOD flag attack-only protos.
- **Copula baseline** for a method comparison (teammate job?).

### What happened on 2026-10-13

- **UNSW connection-type mix failed** (TRY13–16, see submissions table): both variants lose to
  TRY7/TRY11 on *each* seed (paired −0.006 to −0.011). `mix_clean_ratio` and its branch were
  removed from `VAE.ipynb`; UNSW defaults still reproduce `TRY7_UNSW_trim75.csv` byte for byte.
  Negative result: deleting attack-heavy connection types is part of why trim works.
- **NSL temperature sweep done:** 3.0 → 0.93352, 4.0 → 0.93548, 5.0 → 0.93713, 6.0 → 0.93733,
  8.0 → **0.93758 (1st place on public LB)**, plateau. `CONFIGS["NSL"]` is now temperature 8.0 (verified
  byte-identical to `NSL_TRY14_temp8.0.csv`).
- **Why widening works on NSL (measured):** median per-column std of synthetic vs real normals is 0.82 at
  temperature 1 (the VAE is too narrow, a known VAE trait), 1.51 at 4.0, 2.56 at 8.0. Real normals outside the
  synthetic 1–99% range drop from 0.8% to 0.0%, so rare-but-normal rows stop raising false alarms. Filter
  rejection stays low even at temperature 8 (clf 6.4%, ECOD 0.25%), so the filter is not what keeps wide samples
  sane; the gated decoder bounds them.
- **UNSW is the opposite:** TRY7's per-column std is 0.02× real. It flags 23% of real-normal column values
  as out of range vs 42% of attack values. Temperature 2.0 (TRY17) = 0.49337, far below 0.535.
- **UNSW per-column audit of TRY7** (share of values outside the synthetic 1–99% range, normals vs attacks):
  `sbytes`, `dbytes`, `spkts`, `dloss` flag *more normals than attacks* (e.g. `dbytes` 35% vs 28%). They only
  make false alarms. `sttl`/`dttl`/`ct_state_ttl`, `tcprtt`/`synack`/`ackdat`, `ct_srv_*` catch ~3× more attacks
  than normals. This led to per-column trim (see above).
- UNSW TRY17 (TRY7 + temperature 2.0) = 0.49337: widening hurts UNSW badly.

## History: session of 2026-10-12 (its plan is done; superseded numbers kept for the record)

### Best per dataset as of 2026-10-12

| Dataset | Best Kaggle | File (`data/processed/`) | Recipe | Top team |
|---|---|---|---|---|
| UNSW | **0.535** (seed 42; seed 43 = 0.520, so ≈0.527 expected) | `TRY7_UNSW_trim75.csv` | `VAE.ipynb` end to end, UNSW defaults: clf 0.01, d0, sculpt keep 0.35, trim 0.75 | ~0.70 |
| NSL | **0.93287** | `NSL_TRY9_temp2.5.csv` | `VAE.ipynb` end to end, NSL defaults: clf 0.5, d0, **temperature 2.5**, no shaping | 0.936 |

Both best files are produced by running `VAE.ipynb` unchanged (only `DATASET` set). NSL's new
default (temperature 2.5 in `CONFIGS["NSL"]`) was verified to reproduce `NSL_TRY9_temp2.5.csv`
byte for byte.

### What happened on 2026-10-12
- **UNSW** (all `VAE.ipynb`, one change vs TRY7):

  | File | Change | Kaggle |
  |---|---|---|
  | `TRY8_UNSW_trim85.csv` | trim 0.85 | 0.5134 |
  | `TRY9_UNSW_trim90.csv` | trim 0.90 | 0.50738 |
  | `TRY10_UNSW_nosculpt_trim75.csv` | sculpt off | 0.51581 |
  | `TRY11_UNSW_trim75_seed43.csv` | seed 43 | 0.52037 |
  | `TRY12_UNSW_nosculpt_trim75_seed43.csv` | sculpt off + seed 43 | 0.51208 |

- **NSL** (all `VAE.ipynb`, one change vs the 0.9303 baseline):

  | File | Change | Kaggle |
  |---|---|---|
  | `NSL_TRY5_temp1.2.csv` | temperature 1.2 | 0.9306 |
  | `NSL_TRY6_temp1.5.csv` | temperature 1.5 | 0.93117 |
  | `NSL_TRY7_seed43.csv` | seed 43 | 0.9294 |
  | `NSL_TRY8_temp2.0.csv` | temperature 2.0 | 0.93249 |
  | `NSL_TRY9_temp2.5.csv` | temperature 2.5 | **0.93287** |

- **Lesson 1: seed noise.** The same config with a different seed moves the score by about
  **0.015 on UNSW** (0.535 vs 0.520) but only about **0.001 on NSL** (0.9303 vs 0.9294). So
  UNSW differences under about 0.015 from a single submission are not meaningful: the trim
  30–75% sweep is roughly flat-to-rising, not a sharp spike at 75%. 85% and 90% are probably
  really past the peak.
- **Lesson 2: sculpting is worth about +0.013 on UNSW.** Both seeds agree: with sculpt
  0.535 / 0.520 (mean 0.527), without sculpt 0.516 / 0.512 (mean 0.514). Keep `sculpt_keep=0.35`.
- **Lesson 3: NSL likes *wider* data.** Temperature sweep: 0.7 → 0.9280, 1.0 → 0.9303/0.9294,
  1.2 → 0.9306, 1.5 → 0.9312, 2.0 → 0.9325, 2.5 → 0.9329. That's a steady climb, about 3× the
  seed noise, still rising but flattening. The filter (clf 0.5 + ECOD 0.99 quantile on real
  normals) caps the extremes, so higher temperature mostly fills out the normal manifold's
  edges instead of producing junk.
- **Transferability story is now symmetric:** one pipeline with a "tightness" knob. UNSW
  (normals overlap attacks) wants it tight (trim 75% + sculpt), and NSL (normals cleanly
  separated) wants it loose (temperature 2.5, no trim). Ideally a rule that measures class
  overlap sets the direction and strength; at minimum it's a per-dataset config with this evidence.

### Connection-type mix (`mix_clean_ratio`, 2026-10-12; FAILED on Kaggle 2026-10-13, code removed)
- **Finding:** global sculpt + trim decides *which connection types survive* as a side effect.
  - Kaggle TRY7 is 59% `tcp/-/FIN` + 38% `udp/-/CON`.
  - `tcp/http/FIN`, `smtp`, `ftp`, `tcp/-/REQ` are 0%, and `udp/dns/CON` is 2% (real normals: 11.7%).
  - So the detectors flag every missing type. That catches the attack-heavy types (http/FIN is 25%
    of attacks) but also flags about a third of real normals, including clean types like dns/CON
    (0.2% of attacks) and tcp/-/REQ (0%).
  - Caveat: TRY6 (dns 11%) 0.517 < TRY7 (dns 2%) 0.535, which is weak (within noise) evidence
    *against* restoring dns.
- **New config key `mix_clean_ratio`** (stage 6b, default 0 = off; the default reproduces TRY7 byte
  for byte):
  - Types with P(type | attack) / P(type | normal) below the ratio keep their real-normal share and
    are sculpted + trimmed *within* the type.
  - The other types fill the rest with the usual global shaping.
  - `inf` = every type keeps its real share (shaping only within types).
  - Notebook output was cross-checked row for row against the prototype script.
- **Pseudo kaggle holdout S** (UNSW, sculpt 0.35 + trim 0.75, 2 seeds, made from dumped pools):

  | Variant | Seed 42 | Seed 43 | Mean |
  |---|---|---|---|
  | real normals resampled | 0.320 | — | — |
  | TRY7 (global shaping) | 0.393 | 0.423 | 0.408 |
  | restore clean types (ratio 0.1) | 0.426 | 0.414 | 0.420 |
  | drop types with ratio ≥ 1, real share otherwise | 0.428 | 0.408 | 0.418 |
  | real mix for all types (`inf`) | 0.431 | 0.405 | 0.418 |
  | `inf` without trim | 0.410 | 0.390 | 0.400 |
  | sculpt only, no trim (global) | 0.433 | 0.422 | 0.427 |
  | drop types with ratio ≥ 0.5 (also drops `tcp/-/FIN`) | 0.395 | — | — |

  - **Local seed noise is about ±0.015–0.03**, so the mix variants (+0.01 over TRY7) are
    inconclusive locally.
  - Pseudo kaggle also undervalues trim on UNSW (sculpt-only wins locally but lost on Kaggle,
    0.4999 vs 0.527). **Only Kaggle can decide.**

### Plan for 2026-10-13 (done: mix failed, NSL sweep finished)
1. **UNSW: submit 4 files** (all `VAE.ipynb`, UNSW defaults + one change; compare to TRY7/TRY11
   mean ≈ 0.527):

   | File | Change |
   |---|---|
   | `TRY13_UNSW_restore_s42.csv` | `mix_clean_ratio=0.1` (clean types restored to real share) |
   | `TRY14_UNSW_restore_s43.csv` | same, seed 43 |
   | `TRY15_UNSW_mixreal_s42.csv` | `mix_clean_ratio=inf` (real type mix, shaping within types) |
   | `TRY16_UNSW_mixreal_s43.csv` | same, seed 43 |

   - A variant wins only if *both* seeds beat about 0.527 (noise ±0.015).
   - If `inf` wins, it's the best report story: "keep the real connection-type mix, tighten
     values within each type". It needs no attack-aware choice of types.
   - If neither wins, remove the `mix_clean_ratio` key and its `else:` branch in stage 6b, and
     record it as a negative result: "deleting attack-heavy types is part of why trim works".
2. **NSL: submit `NSL_TRY10_temp3.0.csv` and `NSL_TRY11_temp4.0.csv`** (continue the temperature
   sweep; best so far 2.5 = 0.93287). Gains are small; don't spend more than 2.
3. **Pick the final 2 Kaggle submissions per competition** (hidden-test ranking). Current
   leaning: UNSW TRY7 + TRY11 (same config, two seeds), or the mix winner; NSL TRY9 (temp 2.5)
   + the best of the sweep. Don't overfit the public LB: the hidden test has unseen attack types.
4. **Report material is already in the notebooks:**
   - EDA
   - the pseudo kaggle calibration
   - the failure-analysis buckets and ceilings
   - the trim sweep, the seed-noise check, the NSL temperature sweep, the connection-type mix
   - the negative results (option 2 weighting, discrete fix, sculpting on NSL, tightening on NSL,
     temperature on UNSW)
   - The copula baseline (could be a teammate's job) would give a method comparison.

### Previous session (2026-10-05)
- **Sculpt + trim moved into `VAE.ipynb`** (stage 6b), as config keys `sculpt_keep` (1.0 = off) and
  `trim_frac` (0 = off). The filter first collects `N / (keep·(1−trim))` survivors, so every output
  row is a fresh VAE draw (no repeats). The output tag adds `_t{temp}`, `_k{keep}`, `_tr{trim}`
  only when they're non-default.
- UNSW sweep (one change vs TRY3 = sculpt 0.35 + trim 0.30): TRY3 0.51273, TRY4 (+temp 0.7)
  0.51140, TRY5 (trim 0.45) 0.51560, TRY6 (trim 0.60) 0.51659, TRY7 (trim 0.75) 0.535.

### Recipes of the deleted scratch-script files (2026-10-03)
These used sculpt/trim from the scratch script, which resampled kept rows with seed 42 instead of
making fresh rows:
- `synthetic_UNSW_clf0.01_sculpt0.35_quick` (0.4999) = sculpt keep 0.35 (all train attacks) on
  `synthetic_UNSW_clf0.01.csv`.
- `TRY1_UNSW_best_plus_trim30` (0.5135) = drop duplicates of that file, then drop the 30% highest
  ECOD (fit on real normals, one-hot + StandardScaler).
- `TRY2_UNSW_temp0.7_plus_sculpt` (0.5052) = sculpt 0.35 on VAE output (UNSW, clf 0.01, d0,
  temperature 0.7).
- NSL TRY1/TRY3 = trim 30% on the 0.9303 file / on the temp-0.7 file. TRY4a = all 20k real
  normals + 20k random rows of the 0.9303 file, shuffled.

### The big lessons from session 1 (2026-10-03)
1. **Pure fidelity is capped.** Real normals resampled score about 0.93 (NSL) and about
   0.46 (UNSW). UNSW gains must come from shaping the normal data, not from realism.
2. **Attack-aware shaping overfits to seen attack types.** On Kaggle it kept only part of
   the local gain: the filter predicted +15% and got +9%; sculpting predicted +11% and
   got +2%. Kaggle validation contains unseen types that pseudo kaggle can't see.
3. **Label-free tightening helps UNSW and hurts NSL.**
   - UNSW: trim 30% gave +0.014; temperature 0.7 gave +0.005.
   - NSL: trim 30% gave −0.013; temperature 0.7 gave −0.002; both gave −0.015.
   - Reason: UNSW normals overlap with attacks, while NSL normals are already cleanly
     separated, so tightening only removes real variety.
   - So the strength must be a **per-dataset config** (allowed), or set by a rule that
     measures class overlap (better for the transferability grade).
4. **Trust pseudo kaggle selectively.** It tracks Kaggle well on NSL and for attack-aware
   ideas (as a direction, overly optimistic). It is **blind to label-free ideas on UNSW**,
   where trim looked flat locally but won on Kaggle.
5. **On NSL the pure VAE is as good as a real-data mix on Kaggle.** Pseudo kaggle had
   the VAE slightly below real data (0.9286 vs 0.9332) and the 20k real + 20k VAE mix at
   0.9313, but on Kaggle the mix (4a) scored **0.93031**, identical to the pure VAE
   (0.9303). Keep the pure VAE for NSL: it's simpler and a cleaner "generation" story.
   Every NSL variant tried so far lands at or below 0.9303, so NSL looks saturated for
   this approach. The remaining 0.006 gap to the top team is small; don't spend many
   submissions on it.
6. Failure analysis (`src/failure analysis.ipynb`): on UNSW the guards miss about 80% of
   attacks.
   - Misses split into "looks like real normal" (46%, unfixable) and "far from everything
     but individually-normal columns" (54%, the guards' column-by-column blind spot).
   - Our generator is *not* hiding attacks: only 0.2% of misses come from it filling
     empty space.
   - 39% of the guards' top suspicions are real normals with `sttl=254`. Ranking those
     below the attacks would lift S from 0.38 to 0.62 (ceiling).

### Code map
- `src/VAE.ipynb`: the generator (the only pipeline). Per-dataset settings in `CONFIGS`
  (both datasets use `max_discrete_levels=0`, matching all submissions); stage 6b = sculpt + trim.
- `src/pseudo kaggle.ipynb`: local scoring (MODE `all` / `holdout`).
- `src/failure analysis.ipynb`: sections 1–8 with written findings. Needs
  `synthetic_UNSW_clf0.01_holdout.csv`, so keep that file.
- Headless runs: `jupyter nbconvert --execute` on a copy of the notebook whose config cell has
  a `cfg.update(...)` line added after `cfg = {**COMMON, **CONFIGS[DATASET]}`, with `out_path`
  pointed at a scratch dir. Then copy the CSV into `data/processed/` under a new `TRY*` name, so
  submitted files never get overwritten. One UNSW run takes a few minutes; 3–4 can run in parallel.
- Work is **kept local, not committed to git** (user's choice, 2026-10-03).
- `data/processed/` holds every submitted file (see the submissions table) plus
  `synthetic_UNSW_clf0.01_holdout.csv` (needed by the failure analysis) and
  `NSL_TRY4b_real_normals_resampled.csv` (optional real-data-ceiling submission). The user
  decided on 2026-10-05 to keep them for now; superseded ones could be pruned later, since the
  submissions table is the record.

## Submission format (Kaggle)

Competitions / leaderboards:
- NSL-KDD: https://www.kaggle.com/competitions/cs-5344-data-augmentation-for-anomaly-detection-nslkdd/leaderboard
- UNSW-NB15: https://www.kaggle.com/competitions/cs-5344-data-augmentation-for-anomaly-detection-unswnb15/leaderboard

| Dataset | N rows | File |
|---|---|---|
| NSL-KDD | 40,000 | CSV |
| UNSW-NB15 | 70,000 | CSV |

- First column `id` (unique row identifier), then **exactly** the original
  feature columns in original order (categoricals as their original strings).
- No `is_anomaly` / label / prediction column.
- N exceeds the real normal pool (40k vs 20k NSL; 70k vs 35k UNSW), so pure
  resampling of real rows can't fill a submission without duplicates.
- Organizers' detector preprocessing (encoding/scaling) is **unknown**.

## EDA findings (from `src/NSL data_preprocessing.ipynb`, `src/UNSW data_preprocessing.ipynb`)

**Data shape.** Label column `is_anomaly`; categoricals are strings.
- NSL train: 30k rows = 20k normal / 10k anomaly. Normal rows have 69
  `(protocol_type, service, flag)` combos, 40 with <10 rows; top combo
  `tcp/http/SF` = 10.4k. Constant in normal rows: `wrong_fragment`,
  `num_outbound_cmds`, `is_host_login`.
- UNSW train: 50k rows = 35k normal / 15k anomaly. Normal rows have 29
  `(proto, service, state)` combos, 12 with <10 rows; top `tcp/-/FIN` = 17.2k.
  `proto` has 86 values overall but only 6 in normal rows (rare protos like
  `unas`, `sctp` are attack-only).

**Marginals.** Heavy zero-inflation and extreme right skew in both datasets.
One rule for every numeric column (no per-feature thresholds): zero-gate
`P(x=0)` + distribution on `log1p` of the nonzero part. Detect `[0,1]`-bounded
columns by **value range, not name** — NSL's 15 `*_rate` columns have spikes at
both 0 and 1 (3-way gate: 0 / 1 / in-between); UNSW's `rate` is unbounded
packets/sec and UNSW has no continuous bounded columns.

**Correlation blocks** (independent per-column sampling breaks these):
- NSL: root/compromise (`num_compromised`~`num_root` 0.996), serror family,
  rerror family (`rerror_rate`~`srv_rerror_rate` 0.98), `count`~`srv_count`
  (0.94), `same_srv_rate` ≈ 1 − `diff_srv_rate`.
- UNSW: `is_ftp_login`~`ct_ftp_cmd` (0.999), `dpkts`/`dloss`/`dbytes`/`spkts`
  (0.87–0.99), `swin`~`dwin` (0.94), `tcprtt` = `synack` + `ackdat`,
  `ct_srv_src`~`ct_srv_dst` (0.82), `trans_depth`~`ct_flw_http_mthd`.

**Modes.** Normal data is multi-modal and the modes are defined by categorical
combos (NSL: protocol×service×flag; UNSW: mainly proto×state). Unimodal
generators fill the empty space between modes.

**Detectors** (diagnostic only: fit on labeled train, one-hot + StandardScaler,
1 seed × 500 trees):

| Dataset | ECOD AUPRC | IForest AUPRC |
|---|---|---|
| NSL-KDD | 0.774 | 0.601 |
| UNSW-NB15 | 0.391 | 0.362 |

ECOD keys off per-column tails of continuous features (NSL: count/srv-diversity
rates, serror family, `flag_S0`; UNSW: load/bytes/timing/jitter). IForest keys
off one-hot categoricals + count features (NSL: `logged_in`, rerror family,
`flag_REJ`, `service_http`; UNSW: `state_*`, `proto_*`, `swin`/`dwin`, `ct_*`).
→ Synthetic data must get **both** the per-column marginals and the
categorical/joint structure right.

## Current method direction

Conditional VAE in `src/VAE.ipynb` (one config-driven notebook for both datasets):
- Categorical combo sampled from the empirical normal joint → one-hot condition.
- Gated decoder per numeric column (zero/one gates + Gaussian on transformed
  value), mirroring the EDA marginal rules.
- Trained on normal rows only. Anomaly rows are used only after generation: in the filter
  (normal-vs-anomaly classifier + ECOD-on-real-normals manifold check) and in per-column sculpting.
- Stage 6b shaping: `sculpt_keep` (keep the rows least attack-flavoured column by column) then
  `trim_frac` (label-free: drop the rows most outlying under ECOD on real normals).
- Run headless: `.venv/bin/jupyter nbconvert --to notebook --execute <nb> --output-dir <dir>`
  (sed the config line for other datasets/thresholds).
- Outputs go to `data/processed/synthetic_{DATASET}_clf{threshold}[_t][_k][_tr][_holdout].csv`.
  `clf_holdout=True` trains the filter classifier on anomaly half A only, for fair local
  experiments. Submit files made with `clf_holdout=False`. Delete `_holdout` CSVs once a
  sweep's results are recorded here (the user prefers a clean `data/processed/`).
- **Best config = the current `CONFIGS` defaults** (since 2026-10-13): UNSW clf 0.01, d0,
  sculpt 0.35, trim 0.75 (0.535); NSL clf 0.5, d0, temperature 8.0 (0.93758). Everything else
  is the `COMMON` default (seed 42, temperature 1.0, ecod_quantile 0.99). The notebook is
  deterministic.
- `max_discrete_levels=16` (the `COMMON` default, overridden to 0 per dataset to match the
  submissions; scored the same locally): integer columns with ≤16 distinct
  values are a `discrete` type, modelled as a K-way softmax over their observed values,
  so only real values get generated. UNSW: `sttl`, `dttl`, `swin`, `dwin`,
  `ct_state_ttl`, `ct_dst_sport_ltm`, `trans_depth`, `is_ftp_login`, `ct_ftp_cmd`,
  `ct_flw_http_mthd`. NSL: `su_attempted`, `num_failed_logins`, `num_access_files`.
- Keep the notebook simple: failed experiments get recorded here, not left in the code
  as switched-off options.
- Gaussian/vine copula (the EDA notebooks' own leaning) remains the
  alternative/baseline.

## Local evaluation: `src/pseudo kaggle.ipynb`

Local stand-in for the leaderboard. It trains ECOD + IForest (10 seeds × 500
trees) on a synthetic CSV and scores real `train.csv` rows. Encoding is one-hot +
StandardScaler; ordinal encoding gives nearly identical numbers.

- `MODE="all"` evaluates on all train rows. It's calibrated against Kaggle: the
  VAE baseline scored NSL 0.929 locally vs 0.930 on Kaggle, and UNSW 0.496 vs
  0.447. It's optimistic for files whose filter classifier saw the same anomalies.
- `MODE="holdout"` evaluates on train normals + anomaly half B. Shuffle anomalies
  with `random_state=0`; the second half is B, and this must match `VAE.ipynb`.
  Use it to compare anomaly-aware methods. Absolute scores are lower than Kaggle
  (fewer anomalies), so compare files with each other only.
- Blind spot: train anomalies are only `A_seen` types, so gains from targeting
  seen attack types may not transfer to `A_val`/`A_test`.

## Submissions so far

| Date | Dataset | File / config | Public S |
|---|---|---|---|
| 2026-10-03 | NSL | `synthetic_NSL_clf0.5.csv` (VAE baseline) | **0.9303** (top team 0.936) |
| 2026-10-03 | UNSW | `synthetic_UNSW_clf0.5.csv` (VAE baseline) | 0.447 (top team ~0.70) |
| 2026-10-03 | UNSW | `synthetic_UNSW_clf0.01.csv` | **0.489** |
| 2026-10-03 | UNSW | `synthetic_UNSW_clf0.01_sculpt0.35_quick.csv` (0.489 file + per-column sculpting keep 0.35, kept rows repeated; scratch script) | **0.4999** |
| 2026-10-03 | UNSW | `TRY1_UNSW_best_plus_trim30.csv` (best + label-free trim 30%) | **0.5135** |
| 2026-10-03 | UNSW | `TRY2_UNSW_temp0.7_plus_sculpt.csv` (temperature 0.7 + sculpting) | 0.5052 |
| 2026-10-03 | NSL | `NSL_TRY1_trim30.csv` (0.9303 file + label-free trim 30%) | 0.9170 |
| 2026-10-03 | NSL | `NSL_TRY2_temp0.7.csv` (VAE temperature 0.7, clf 0.5, `max_discrete_levels=0`) | 0.9280 |
| 2026-10-03 | NSL | `NSL_TRY3_temp0.7_trim30.csv` (temperature 0.7 + trim 30%) | 0.9153 |
| 2026-10-03 | NSL | `NSL_TRY4a_mix_real20k_vae20k.csv` (all 20k real normals + 20k rows of the 0.9303 file) | 0.93031 (same as base) |
| 2026-10-05 | UNSW | `TRY3_UNSW_pipeline_repro.csv` (`VAE.ipynb` end to end: clf 0.01, d0, sculpt 0.35, trim 0.30, fresh rows) | 0.51273 |
| 2026-10-05 | UNSW | `TRY4_UNSW_temp0.7.csv` (TRY3 + temperature 0.7) | 0.51140 |
| 2026-10-05 | UNSW | `TRY5_UNSW_trim45.csv` (TRY3 with trim 0.45) | 0.51560 |
| 2026-10-05 | UNSW | `TRY6_UNSW_trim60.csv` (TRY3 with trim 0.60) | 0.51659 |
| 2026-10-05 | UNSW | `TRY7_UNSW_trim75.csv` (TRY3 with trim 0.75; now the notebook default) | **0.535** |
| 2026-10-12 | UNSW | `TRY8_UNSW_trim85.csv` (TRY7 with trim 0.85) | 0.5134 |
| 2026-10-12 | UNSW | `TRY9_UNSW_trim90.csv` (TRY7 with trim 0.90) | 0.50738 |
| 2026-10-12 | UNSW | `TRY10_UNSW_nosculpt_trim75.csv` (TRY7 with sculpt off) | 0.51581 |
| 2026-10-12 | UNSW | `TRY11_UNSW_trim75_seed43.csv` (TRY7 with seed 43) | 0.52037 |
| 2026-10-12 | UNSW | `TRY12_UNSW_nosculpt_trim75_seed43.csv` (TRY10 with seed 43) | 0.51208 |
| 2026-10-12 | NSL | `NSL_TRY5_temp1.2.csv` (baseline + temperature 1.2) | 0.9306 |
| 2026-10-12 | NSL | `NSL_TRY6_temp1.5.csv` (temperature 1.5) | 0.93117 |
| 2026-10-12 | NSL | `NSL_TRY7_seed43.csv` (baseline with seed 43) | 0.9294 |
| 2026-10-12 | NSL | `NSL_TRY8_temp2.0.csv` (temperature 2.0) | 0.93249 |
| 2026-10-12 | NSL | `NSL_TRY9_temp2.5.csv` (temperature 2.5; now the notebook default) | **0.93287** |
| 2026-10-13 | UNSW | `TRY13_UNSW_restore_s42.csv` (TRY7 + `mix_clean_ratio=0.1`) | 0.52829 |
| 2026-10-13 | UNSW | `TRY14_UNSW_restore_s43.csv` (TRY13 with seed 43) | 0.51443 |
| 2026-10-13 | UNSW | `TRY15_UNSW_mixreal_s42.csv` (TRY7 + `mix_clean_ratio=inf`) | 0.52920 |
| 2026-10-13 | UNSW | `TRY16_UNSW_mixreal_s43.csv` (TRY15 with seed 43) | 0.50902 |
| 2026-10-13 | NSL | `NSL_TRY10_temp3.0.csv` (temperature 3.0) | 0.93352 |
| 2026-10-13 | NSL | `NSL_TRY11_temp4.0.csv` (temperature 4.0) | 0.93548 |
| 2026-10-13 | NSL | `NSL_TRY12_temp5.0.csv` (temperature 5.0) | 0.93713 |
| 2026-10-13 | NSL | `NSL_TRY13_temp6.0.csv` (temperature 6.0) | 0.93733 |
| 2026-10-13 | NSL | `NSL_TRY14_temp8.0.csv` (temperature 8.0) | **0.93758** (1st place on public LB) |
| 2026-10-13 | UNSW | `TRY17_UNSW_temp2.0.csv` (TRY7 + temperature 2.0) | 0.49337 |
| 2026-10-14 | UNSW | `TRY18_UNSW_percol_sep_s42.csv` (TRY7 + per-column trim, `sep` 0.2) | 0.52405 |
| 2026-10-14 | UNSW | `TRY19_UNSW_percol_sep_s43.csv` (TRY18 with seed 43) | 0.5033 |
| 2026-10-14 | UNSW | `TRY22_UNSW_trimIF_s42.csv` (TRY7 + `trim_by="ecod+iforest"`) | 0.52455 |
| 2026-10-14 | UNSW | `TRY23_UNSW_trimIF_s43.csv` (TRY22 with seed 43) | 0.52227 |

## Leaderboard learnings (2026-10-03)

- Public scores, VAE baseline (`clf_threshold=0.5`): **NSL 0.9303** (team best;
  top team 0.936), **UNSW 0.447** (top team ~0.70). The same team leads both, so
  their method probably transfers across datasets.
- **Fidelity has hit its ceiling.** Real normal rows resampled to N score about
  0.933 on NSL and about 0.465 on UNSW (`all` mode). A perfect copy of the normal
  distribution can't reach 0.7 on UNSW, so the gap has to come from deliberately
  shaping the normal data using the anomaly rows (utility over fidelity).
- **Pruning attack-like normals helps UNSW.** Experiment on real normals, scored
  in holdout mode: drop normals whose out-of-fold P(anomaly) from a classifier is
  ≥ t. UNSW S went from 0.324 (no pruning) to 0.399 at t=0.01, which drops 28%
  of normals. No effect on NSL, where normals almost never look like attacks.
  Mechanism: normal rows in attack-heavy regions teach the detectors that the
  region is normal. Removing them makes attacks there land in empty space and
  score high, at the cost of false alarms on real normals there.
- **Trimming generic outliers doesn't help.** Dropping the most ECOD-outlying
  normals (label-free) gave UNSW 0.324 → 0.350 at best, and made NSL worse.
  Shrinking "normal" in general is not the lever; removing attack-like regions is.
- `clf_threshold` in `VAE.ipynb` is the knob for this (it drops generated rows
  with P(anomaly) ≥ threshold). The default 0.5 is far too loose; gains start
  below about 0.1.
- VAE `clf_threshold` sweep (holdout mode, S). UNSW: 0.5 → 0.333, 0.2 → 0.356,
  0.1 → 0.376, 0.05 → 0.381, **0.01 → 0.383** (real normals score 0.320).
  NSL: 0.5 → 0.906, 0.05 → 0.907 (no gain; real normals score 0.911).
  At 0.01 the UNSW filter rejects about 34% of candidates.
  **Kaggle: UNSW clf0.01 = 0.489** (vs 0.447 at 0.5). That's +9% on the leaderboard vs
  +15% predicted by pseudo kaggle. Expected, since the leaderboard includes unseen `A_val` types. Pruning alone
  probably won't reach 0.7.
- **Option 2, weighted training, tried and gave no gain (code removed).** Normal rows are
  weighted by (1 − out-of-fold P(anomaly))^γ for minibatches and combo sampling,
  with clf 0.01, holdout mode. UNSW S: γ=0 0.378, γ=10 0.382, γ=50 0.381,
  γ=200 0.375 (noise level). NSL: γ=0 0.907, γ=50 0.907.
  The filter's rejection rate falls from 30% to under 1% at γ=50, so the VAE
  generates clean rows directly. But the final distribution, and so the score,
  is the same as filtering after the fact. Both approaches remove the same
  attack-like region (e.g. UNSW `udp/-/INT` drops from 5.2% to about 0%).
  Conclusion: "remove attack-like normals" plateaus at about 0.38 holdout (about
  0.49 Kaggle). Reaching 0.7 needs a different lever.

## Failure analysis: `src/failure analysis.ipynb` (UNSW, 2026-10-03)

Copies the guards (ECOD + IForest 10×500) trained on
`synthetic_UNSW_clf0.01_holdout.csv` and scores train normals + anomaly half B. That
holdout file is kept in `data/processed/` on purpose, because the notebook needs it.
A "miss" = an attack scored below the 95th percentile of normal scores. Each miss
gets a bucket by nearest-neighbour distance in standardised one-hot space
(r = 90th percentile of normal→nearest-other-normal distance):

- The guards miss most attacks: ECOD 85%, IForest 81%, both 78%.
- **B (close to synthetic only, i.e. our generator filled empty space): 0.2% of
  misses.** Perfectly fixing it gives S 0.383 → 0.386. Pruning and filtering ideas
  are exhausted, which explains the flat threshold and weighting results.
- **A (sits on real normal traffic): 46%.** No normal-only dataset can fix these;
  mostly `tcp/-/FIN` and `tcp/http/FIN`.
- **C (far from both real and synthetic, yet scored low): 54%.** Perfectly fixing C
  gives S 0.383 → **0.750**, about the top team's level. These attacks have every
  individual column inside the normal central range (`dpkts`, `dbytes`, `sbytes`,
  `ct_*`, `ct_state_ttl`), but in an abnormal combination. ECOD (scores each column
  separately) and IForest (cuts one column at a time) are both blind to that.
- **Lever for the remaining gap:** reshape what the guards see column by column, so
  bucket-C combinations become rare in at least one column. Ideas: change value
  distributions *within* each categorical combo, or tighten clusters so the gaps
  between them are emptier. More realism or harder pruning won't help.
- Not yet run for NSL. It would need a `synthetic_NSL_clf0.5_holdout.csv` first.
- **Section 7, looking inside each connection type:**
  - For `tcp/smtp/FIN` and `udp/dns/INT` (about 20% of C misses), the TTL columns
    separate attacks 100% within the type. smtp attacks have `sttl` 62/254, `dttl` 252
    and `ct_state_ttl` 1; smtp normals have `sttl` 31, `dttl` 29 and `ct_state_ttl` 0.
    But 27% of all normals have `sttl=254`, so the guards, which mix all types together,
    miss them.
  - Across the dataset TTL is nearly an attack fingerprint (attacks: `sttl` 254 or 62;
    normals: 65% are 31).
  - Our clf0.01 synthetic data already dropped TTL=254 (`sttl` ≈ 31, `ct_state_ttl`=0
    in 96% of rows), which is the likely source of the 0.447 → 0.489 gain. The remaining
    cost is that real normals with TTL=254 now look suspicious too and crowd the top of
    the ranking.
  - For `tcp/http/FIN` and `tcp/-/FIN` (about 59% of C misses), no column separates them
    even within the type. These are genuine combination anomalies, hard for
    column-by-column guards.
  - **VAE flaw:** few-valued integer columns get smeared (synthetic `sttl` takes 29/30/32,
    which never occur in real data). Fix: model low-cardinality integer columns as
    categories in the decoder.
  - Next levers: (1) that decoder fix; (2) among TTL=254 rows, find what separates real
    normals from attacks.
- **Decoder fix done (`max_discrete_levels=16`): it fixes realism but doesn't change the
  score.** Synthetic `sttl` is now only real values (31: 95%, 254: 3%, 62: 1%), where it
  used to include smeared 29/30/32. Pseudo kaggle holdout: UNSW 0.383 → 0.386,
  NSL 0.906 → 0.906, both within noise (about ±0.004). Kept because it's a more
  correct and more explainable model, not for the score.
- **Section 8, false alarms (UNSW):** 39% of the guards' 7,500 most-suspicious rows are
  real normals with `sttl=254`, because our synthetic data has only 0.1% `sttl=254`.
  Ceiling: if they were ranked below all attacks, S would go from 0.383 to **0.618**,
  the biggest chunk of lost score found. But no column separates 254-normals from
  254-attacks well (best `ct_dst_src_ltm` at 0.33; normals have slightly higher
  `ct_*` counts). Connection type helps a little: `tcp/-/REQ` is 7.8% of 254-normals
  vs 0% of 254-attacks, and `tcp/http/FIN` is 1.8% vs 18%. Re-adding 254 in bulk hurts
  (that's what looser `clf_threshold` did), so any re-adding must be selective.
- **Per-column sculpting (tested 2026-10-03, scratch script, not yet in the pipeline).**
  This drops rows by a *per-column* attack likelihood ratio rather than the whole-row
  classifier, to match how the guards judge (one column at a time):
  - Each column is cut into 20 quantile bins (or uses its raw values / categories if it has
    few). Per bin, LLR = log P(bin | attack half A) − log P(bin | normal), with Laplace
    smoothing.
  - Row score = Σ_columns max(0, LLR). Keep the lowest-scoring fraction `keep`.
  - Results, pseudo kaggle holdout S:

    | Pool | Setting | Whole-row referee | Per-column sculpting |
    |---|---|---|---|
    | UNSW VAE output (0.489 version) | keep 1.0 | 0.384 | — |
    | | keep 0.4 | 0.411 | **0.426** |
    | | keep 0.3 | 0.403 | **0.430** |
    | | keep 0.2 | 0.408 | **0.431** |
    | | keep 0.1 | 0.391 | **0.440** |
    | UNSW real normals | keep 0.55 | 0.405 | 0.411 |
    | NSL real normals | keep 1.0 | 0.911 | — |
    | | keep 0.7 | 0.903 | **0.868** |
    | | keep 0.4 | 0.899 | **0.874** |

  - **It hurts NSL** (0.911 → 0.87), so `keep` must be a per-dataset config (NSL 1.0 =
    off). Better still, choose it automatically with an internal half-A/B validation, so
    the rule stays transferable.
  - Caveat: keep 0.1–0.2 means very aggressive specialisation to seen attack types.
    Kaggle has previously given about 60% of the pseudo-kaggle gain, so prefer a
    moderate keep (0.3–0.4). In the pipeline, oversample VAE candidates by 1/keep
    rather than duplicating kept rows.
  - **Kaggle check: 0.489 → 0.4999 (+0.011).** Pseudo kaggle predicted 0.384 → 0.428
    (+11%), but Kaggle gave only about 20% of that. Attack-aware shaping transfers less
    and less as it gets more aggressive, which points to overfitting to `A_seen`.
  - **Key insight:** pseudo kaggle only contains seen attack types, so it is *biased toward
    attack-aware methods* and blind to how label-free ideas do on unseen types. Ideas that
    looked flat locally (e.g. label-free trimming, lower temperature) were never tested on
    Kaggle, the only place with unseen types.
