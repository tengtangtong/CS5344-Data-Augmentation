import sys; sys.path.insert(0,'.')
from lib import *
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
syn=pd.read_csv(f"{ROOT}/data/processed/TRY7_UNSW_trim75.csv")
ev=df.copy(); y=ev[L].to_numpy(); K=int(y.sum())
A=pd.get_dummies(syn[feature_cols],columns=cat_cols,dtype=float)
B=pd.get_dummies(ev[feature_cols],columns=cat_cols,dtype=float).reindex(columns=A.columns,fill_value=0.0)
sc=StandardScaler().fit(A); A,B=sc.transform(A),sc.transform(B)
scores={"ECOD":ECOD().fit(A).decision_function(B),
        "IForest":np.mean([pd.Series(IForest(n_estimators=500,random_state=s).fit(A).decision_function(B)).rank().to_numpy() for s in range(3)],axis=0)}
ev["combo"]=ev.proto+"/"+ev.service+"/"+ev.state; syn["combo"]=syn.proto+"/"+syn.service+"/"+syn.state
nrm=ev[L]==0
lo,hi=syn[num_cols:=[c for c in feature_cols if c not in cat_cols]].quantile(0.01),syn[num_cols].quantile(0.99)
out=(ev[num_cols]<lo)|(ev[num_cols]>hi)
for det,s in scores.items():
    top=np.zeros(len(ev),bool); top[np.argsort(-s)[:K]]=True
    fa=top&nrm
    print(f"\n===== {det}: AUPRC {average_precision_score(y,s):.3f} | top {K}: {fa.sum()} real normals ({fa.sum()/K:.0%}) | catches {(top&~nrm).sum()/K:.0%} of attacks")
    f254=fa&(ev.sttl==254); print(f"  false alarms with sttl=254: {f254.sum()/fa.sum():.0%}   other: {(fa&~f254).sum()}")
    o=fa&(ev.sttl!=254)
    print("  non-254 false alarms by connection type (share | real-normal share | synth share | type's normals flagged):")
    t=pd.DataFrame({"fa":ev[o].combo.value_counts(normalize=True),"real":ev[nrm].combo.value_counts(normalize=True),
                    "synth":syn.combo.value_counts(normalize=True)}).fillna(0)
    t["flag_rate"]=(ev[fa].combo.value_counts()/ev[nrm].combo.value_counts()).reindex(t.index).fillna(0)
    print(t.sort_values("fa",ascending=False).head(8).round(3).to_string())
    print("  columns most often out of synthetic 1-99% range in non-254 false alarms (vs caught attacks):")
    c=pd.DataFrame({"false_alarms":out[o].mean(),"caught_attacks":out[top&~nrm].mean(),"all_normals":out[nrm].mean()})
    print(c.sort_values("false_alarms",ascending=False).head(10).round(2).to_string())
print("\n#3 rare types: real-normal share vs synthetic share vs flag rate (IForest top-K)")
s=scores["IForest"]; top=np.zeros(len(ev),bool); top[np.argsort(-s)[:K]]=True
t=pd.DataFrame({"real_n":ev[nrm].combo.value_counts(),"real_share":ev[nrm].combo.value_counts(normalize=True),
                "synth_share":syn.combo.value_counts(normalize=True),
                "flag_rate":(ev[top&nrm].combo.value_counts()/ev[nrm].combo.value_counts()),
                "attack_n":ev[~nrm].combo.value_counts()}).fillna(0)
t=t[t.real_n>=20].sort_values("real_n",ascending=False)
print(t.round(3).to_string())
