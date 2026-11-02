# Leave-cluster-out pseudo kaggle: does a local evaluator with "unseen" attack types rank past
# UNSW submissions like Kaggle did?  Usage: python loco.py <seed> <fold> <out.csv>
#   fold "std"  = current pseudo kaggle holdout (method sees attack half A, eval = normals + half B)
#   fold 0..4   = hide attack cluster k entirely; method sees half A of the other clusters,
#                 eval = normals + half B of the other clusters + all of cluster k
import sys; sys.path.insert(0, '.')
from lib import *
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score

seed, fold, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
N_FINAL, IF_SEEDS = 20_000, 3

# --- attack pseudo-types: KMeans(6) on log1p numerics + one-hot cats; big clusters only are held out
anom_all = df[df[L] == 1].copy()
num = [c for c in feature_cols if c not in cat_cols]
Xa = pd.concat([np.log1p(anom_all[num].clip(lower=0)), pd.get_dummies(anom_all[cat_cols], dtype=float)], axis=1)
lab = KMeans(6, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Xa))
sizes = pd.Series(lab).value_counts()
big = sorted(sizes[sizes >= 1000].index, key=lambda k: -sizes[k])   # the 5 big clusters, largest first
anom_all["cl"] = lab
if fold == "std":
    unseen = anom_all.iloc[:0]; seen = anom_all
else:
    k = big[int(fold)]; unseen = anom_all[anom_all.cl == k]; seen = anom_all[anom_all.cl != k]
seen = seen.sample(frac=1, random_state=0)
sA, sB = seen.iloc[:len(seen) // 2], seen.iloc[len(seen) // 2:]
eval_df = pd.concat([normal_df, sB, unseen], ignore_index=True); y = eval_df[L].to_numpy()

clf = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed)
tr = pd.concat([normal_df, sA[feature_cols + [L]]], ignore_index=True)
clf.fit(clf_matrix(tr), tr[L])

def pk(synth):
    A = pd.get_dummies(synth[feature_cols], columns=cat_cols, dtype=float)
    B = pd.get_dummies(eval_df[feature_cols], columns=cat_cols, dtype=float).reindex(columns=A.columns, fill_value=0.0)
    sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
    e = average_precision_score(y, ECOD().fit(A).decision_function(B))
    f = np.mean([average_precision_score(y, IForest(n_estimators=500, random_state=s).fit(A).decision_function(B))
                 for s in range(IF_SEEDS)])
    return e, f

# (name, temperature, clf_threshold, sculpt_keep, trim_frac, trim_by, kaggle)
CONFIGS = {42: [
    ("clf0.5",          1.0, 0.5,  1.0,  0.0,  "ecod", 0.447),
    ("clf0.01",         1.0, 0.01, 1.0,  0.0,  "ecod", 0.489),
    ("sculpt",          1.0, 0.01, 0.35, 0.0,  "ecod", 0.4999),
    ("TRY3 trim.30",    1.0, 0.01, 0.35, 0.30, "ecod", 0.51273),
    ("TRY5 trim.45",    1.0, 0.01, 0.35, 0.45, "ecod", 0.5156),
    ("TRY6 trim.60",    1.0, 0.01, 0.35, 0.60, "ecod", 0.51659),
    ("TRY7 trim.75",    1.0, 0.01, 0.35, 0.75, "ecod", 0.535),
    ("TRY8 trim.85",    1.0, 0.01, 0.35, 0.85, "ecod", 0.5134),
    ("TRY9 trim.90",    1.0, 0.01, 0.35, 0.90, "ecod", 0.50738),
    ("TRY10 nosculpt",  1.0, 0.01, 1.0,  0.75, "ecod", 0.51581),
    ("TRY22 trimIF",    1.0, 0.01, 0.35, 0.75, "ecod+iforest", 0.52455),
    ("TRY4 t0.7",       0.7, 0.01, 0.35, 0.30, "ecod", 0.5114),
    ("TRY17 t2.0",      2.0, 0.01, 0.35, 0.75, "ecod", 0.49337),
], 43: [
    ("TRY11 s43",       1.0, 0.01, 0.35, 0.75, "ecod", 0.52037),
    ("TRY12 s43 nosc",  1.0, 0.01, 1.0,  0.75, "ecod", 0.51208),
    ("TRY23 s43 trimIF",1.0, 0.01, 0.35, 0.75, "ecod+iforest", 0.52227),
]}

cands = {}
def get_cands(temp):
    if temp not in cands:
        cands.clear()
        f = f"cands_s{seed}.pkl" if temp == 1.0 else f"cands_s{seed}_t{temp}.pkl"
        c = pd.read_pickle(f).drop(columns=["p_anom"])
        c["pA"] = clf.predict_proba(clf_matrix(c))[:, 1]
        cands[temp] = c
    return cands[temp]

iforest_trim = None
rows = []
for name, temp, thr, keep, trim, trim_by, kag in CONFIGS[seed]:
    c = get_cands(temp)
    need = int(np.ceil(N_FINAL / (keep * (1 - trim))))
    pool = c[(c.pA < thr) & (c.ecod <= ecod_cut)]
    if len(pool) < need:
        print("SKIP", name, "pool", len(pool), "need", need, flush=True); continue
    pool = pool.iloc[:need].reset_index(drop=True)
    if keep < 1:
        pool = keep_lowest(pool, sculpt_score(pool, normal_df, sA), keep)
    if trim > 0:
        Xp = scaler.transform(onehot_matrix(pool)); sc_ = ecod.decision_function(Xp)
        if trim_by == "ecod+iforest":
            if iforest_trim is None:
                iforest_trim = IForest(n_estimators=500, random_state=seed).fit(scaler.transform(onehot_matrix(normal_df)))
            sc_ = pd.Series(sc_).rank().to_numpy() + pd.Series(iforest_trim.decision_function(Xp)).rank().to_numpy()
        pool = keep_lowest(pool, sc_, 1 - trim)
    synth = pool.iloc[:N_FINAL]
    e, f = pk(synth)
    rows.append(dict(seed=seed, fold=fold, config=name, kaggle=kag, ECOD=e, IForest=f, S=(e + f) / 2,
                     n_eval_anom=int(y.sum()), n_unseen=len(unseen)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
