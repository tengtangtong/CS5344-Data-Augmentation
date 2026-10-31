"""Does the ECOD combination rule explain the proxy–Kaggle mismatch?

Scores UNSW-NB15 files that already have Kaggle public scores on the released
training rows (in-sample, so absolute values are optimistic) with
  old  : our fallback  O = max(U_skew, (U_l + U_r) / 2)
  pyod : PyOD source   O = max(U_skew, max(U_l, U_r))
and reports the Spearman correlation of each with the public scores."""
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.metrics import average_precision_score as ap

from pipeline1 import _ecdf_cols, encode, iforest_scores

KAGGLE = {
    "kappa_ladder/submission_UNSW-NB15_p16_k4_d10.csv": 0.55102,
    "p16_candidates/submission_UNSW-NB15_p16_k6_d10.csv": 0.55427,
    "kappa_ladder/submission_UNSW-NB15_p16_k50_d10.csv": 0.56135,
    "kappa_ladder/submission_UNSW-NB15_p16_k100_d10.csv": 0.56049,
    "kappa_ladder/submission_UNSW-NB15_p16_k1000_d10.csv": 0.56103,
    "kappa_ladder/submission_UNSW-NB15_p16_k50_d07.csv": 0.55422,
    "kappa_ladder/submission_UNSW-NB15_p16_k50_d15.csv": 0.53118,
    "FINAL/submission_UNSW-NB15_final.csv": 0.56335,
    "FINAL/pipeline17/submission_UNSW-NB15_p17.csv": 0.56447,
    "overfit_probes/submission_UNSW-NB15_P1_seenfit_tilt_wide.csv": 0.55279,
    "overfit_probes/submission_UNSW-NB15_P2_seenfit_greedy_neutralise.csv": 0.54161,
    "overfit_probes/submission_UNSW-NB15_P3_seenfit_hardtrim.csv": 0.55386,
}
cats = ["proto", "service", "state"]
test = pd.read_csv("../train_UNSW-NB15.csv")
for c in cats:
    test[c] = test[c].astype(str)
y = test.pop("is_anomaly").to_numpy()
nums = [c for c in test.columns if c not in cats]


def ecod(Xtr, Xte, rule):
    X = np.vstack([Xtr, Xte]).astype(float)
    U_l, U_r = -np.log(_ecdf_cols(X)), -np.log(_ecdf_cols(-X))
    with np.errstate(all="ignore"):
        sk = np.nan_to_num(np.sign(skew(X, axis=0)))
    U_skew = U_l * -1 * np.sign(sk - 1) + U_r * np.sign(sk + 1)
    other = (U_l + U_r) / 2 if rule == "old" else np.maximum(U_l, U_r)
    return np.maximum(U_skew, other).sum(1)[-len(Xte):]


rows = []
for f, kag in KAGGLE.items():
    syn = pd.read_csv(f).drop(columns=["id"])
    for c in cats:
        syn[c] = syn[c].astype(str)
    r = dict(file=f.split("/")[-1].replace("submission_UNSW-NB15_", ""), kaggle=kag)
    for mode in ["onehot", "ordinal"]:
        A, B = encode(syn, test, nums, cats, mode)
        f_if = ap(y, iforest_scores(A, B, 3, 200))
        for rule in ["old", "pyod"]:
            r["{}_{}".format(rule, mode)] = (ap(y, ecod(A, B, rule)) + f_if) / 2
            r["ecod_{}_{}".format(rule, mode)] = ap(y, ecod(A, B, rule))
        r["iforest_" + mode] = f_if
    rows.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
tab = pd.DataFrame(rows)
tab.to_csv("check_ecod_rule.csv", index=False)
print("\nSpearman correlation with Kaggle public score (n = {}):".format(len(tab)))
for c in [c for c in tab.columns if c not in ("file", "kaggle")]:
    print("  {:<22} {:+.3f}".format(c, spearmanr(tab[c], tab["kaggle"]).correlation))
