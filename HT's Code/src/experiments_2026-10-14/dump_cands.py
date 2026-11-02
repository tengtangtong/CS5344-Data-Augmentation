# Exec VAE.ipynb cells (UNSW, given seed) up to the filter models, then dump raw candidates + scores.
import json, sys, matplotlib; matplotlib.use("Agg")
seed = int(sys.argv[1]); out = sys.argv[2]
nb = json.load(open("/Users/han./Desktop/CS5344/src/VAE.ipynb"))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
g = {}
src0 = code[0].replace('DATASET = "NSL"', 'DATASET = "UNSW"').replace(
    'cfg = {**COMMON, **CONFIGS[DATASET]}', f'cfg = {{**COMMON, **CONFIGS[DATASET]}}; cfg["seed"] = {seed}')
exec(src0, g)
for c in code[1:4]: exec(c, g)          # specs, model, training
exec(code[4].replace("demo = generate(5)\ndemo", ""), g)
# filter models only (stop before the loop)
exec(code[5].split("# 6b drops rows later")[0], g)
import pandas as pd, numpy as np
parts = []
for _ in range(6):
    cand = g["generate"](300_000)
    cand["p_anom"] = g["clf"].predict_proba(g["clf_matrix"](cand))[:, 1]
    cand["ecod"] = g["ecod"].decision_function(g["scaler"].transform(g["onehot_matrix"](cand)))
    parts.append(cand)
pd.concat(parts, ignore_index=True).to_pickle(out)
print("ecod_cut", g["ecod_cut"])
