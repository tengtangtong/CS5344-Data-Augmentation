import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD
ROOT="/Users/han./Desktop/CS5344"
df=pd.read_csv(f"{ROOT}/data/raw/UNSW-NB15_train.csv"); L="is_anomaly"
cat_cols=["proto","service","state"]; feature_cols=[c for c in df.columns if c!=L]
normal_df=df[df[L]==0].reset_index(drop=True)
anom=df[df[L]==1].sample(frac=1,random_state=0); anom_A=anom.iloc[:len(anom)//2]; anom_B=anom.iloc[len(anom)//2:]
nb=json.load(open(f"{ROOT}/src/VAE.ipynb")); code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g=dict(np=np,pd=pd,feature_cols=feature_cols,cat_cols=cat_cols)
exec(code[6].split("attacks = clf_train")[0], g)   # sculpt_score, keep_lowest
sculpt_score, keep_lowest = g["sculpt_score"], g["keep_lowest"]
all_levels={c:sorted(df[c].unique()) for c in cat_cols}
def clf_matrix(d):
    X=d[feature_cols].copy()
    for c in cat_cols: X[c]=pd.Categorical(X[c],categories=all_levels[c])
    return X
ohe_cols=pd.get_dummies(normal_df[feature_cols],columns=cat_cols,dtype=float).columns
def onehot_matrix(d): return pd.get_dummies(d[feature_cols],columns=cat_cols,dtype=float).reindex(columns=ohe_cols,fill_value=0.0).to_numpy()
scaler=StandardScaler().fit(onehot_matrix(normal_df)); ecod=ECOD().fit(scaler.transform(onehot_matrix(normal_df)))
ecod_cut=np.quantile(ecod.decision_scores_,0.99)
def ecod_score(d): return ecod.decision_function(scaler.transform(onehot_matrix(d)))
