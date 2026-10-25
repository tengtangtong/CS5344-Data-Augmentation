"""Extra robustness check for Pipeline 17: score the learned plan against the
Pipeline 16 plan on further independent behaviour splits (new seeds and
rotations). Reads the learned plan from pipeline17_output/<dataset>_p17_metadata.json.

Usage: python p17_robustness.py UNSW-NB15 3 4
"""
import json
import sys
from pathlib import Path

import pandas as pd

import pipeline17 as p17
from pipeline16 import DATASETS, load_data, settings_for

name, rotations = sys.argv[1], [int(r) for r in sys.argv[2:]] or [3, 4]
cfg = next(c for c in DATASETS if c["name"] == name)
meta = json.loads((p17.OUTDIR / "{}_p17_metadata.json".format(name)).read_text())
S = settings_for(name)
df, feats, nums = load_data(cfg)
rows = []
for r in rotations:
    sp = p17.Split(df, feats, nums, cfg["cats"], name, S, p17.GENERATOR[name], cfg["seed"] + 5000 + 97 * r, rotation=r)
    for tag, plan, a, rho in [("Pipeline 16", p17.p16_plan(sp.ctx, S), 0.5, 0.0),
                              ("Pipeline 17", meta["kappa_plan"], meta["alpha"], meta["rho"])]:
        m = sp.score(plan, a, rho)
        rows.append(dict(rotation=r, hidden=str(sp.meta["hidden_types"]), novel_share=sp.meta["hidden_share"],
                         variant=tag, seen=m["seen"], pseudo_novel=m["unseen"]))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
tab.to_csv(p17.OUTDIR / "{}_p17_robustness.csv".format(name), index=False)
