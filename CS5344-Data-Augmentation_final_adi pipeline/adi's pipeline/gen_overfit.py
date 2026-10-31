"""Seen-type overfitting probes for UNSW-NB15 (all tuned ONLY on seen attack types).
P1 tilt_wide : per-column quantile tilt, wide kappa grid, 3 passes (seen-type validation)
P2 greedy    : per-column keep / spread(k=8) / const(mode), greedy on seen-type validation
P3 hardtrim  : classifier pushed to the extreme — pre-trim 25%, gamma=512, keep 5% of candidates
P1/P2 are applied to the same candidate pool as the 0.55427 (kappa=6) file."""
import json, numpy as np, pandas as pd, pipeline1 as P
from sklearn.metrics import average_precision_score as ap
from pathlib import Path
out = Path("overfit_probes"); out.mkdir(exist_ok=True)
cfg = P.DATASETS[1]; name = cfg["name"]
df, feats, nums = P.load_data(cfg); cats = cfg["cats"]
n = df[df[P.LABEL] == 0][feats].reset_index(drop=True); a = df[df[P.LABEL] == 1][feats].reset_index(drop=True)
S = P.settings_for(name, spread_kappa=6.0)
log = {}

# ---------- P1: wide tilt plan
plan_t = P.learn_tilt_plan(df, feats, nums, cats, name, S, cfg["seed"],
                           kappas=(0.2, 0.33, 0.5, 2.0, 3.0, 5.0), passes=3)
log["P1_tilt_plan"] = plan_t; print("P1 plan", plan_t, flush=True)

# ---------- P2: greedy keep/spread/const on seen-type validation
rng = np.random.default_rng(cfg["seed"]); vm = rng.random(len(df)) < 0.3
fit, val = df[~vm], df[vm]
nf = fit[fit[P.LABEL] == 0][feats].reset_index(drop=True); af = fit[fit[P.LABEL] == 1][feats].reset_index(drop=True)
cv = P.prepare("vae", nf, af, feats, nums, cats, name, S, cfg["seed"], 20000, overgen_max=20)
b0, _ = P.finalize(cv, S, 20000, cfg["seed"]); yv = val[P.LABEL].to_numpy()
modes = {c: float(nf[c].mode().iloc[0]) for c in nums}
def apply_ops(syn, spec, plan):
    s = syn.copy(); sp = [c for c, o in plan.items() if o == "spread"]
    if sp: s = P.apply_spread(s, spec, sp, 8.0)[feats]
    for c, o in plan.items():
        if o == "const": s[c] = modes[c]
    return P.restore_derived(s, spec)[feats]
def sc(syn):
    A, B = P.encode(syn, val, nums, cats, "onehot")
    return (ap(yv, P.ecod_scores(A, B)) + ap(yv, P.iforest_scores(A, B, 2, 100))) / 2
derived = [d["target"] for d in cv["spec"]["derived"]]
plan_g, best = {}, sc(b0); print("P2 start val", round(best, 4), flush=True)
for step in range(25):
    trials = [(sc(apply_ops(b0, cv["spec"], dict(plan_g, **{c: o}))), c, o)
              for c in nums if c not in plan_g and c not in derived for o in ("spread", "const")]
    s_, c, o = max(trials)
    if s_ <= best + 0.001: break
    plan_g[c] = o; best = s_; print(f"  P2 step {step+1}: {o} {c} val={best:.4f}", flush=True)
log["P2_greedy_plan"] = plan_g; log["P2_val"] = best
del cv

# ---------- apply P1 / P2 to the full-size kappa=6 pool
ctx = P.prepare("vae", n, a, feats, nums, cats, name, S, cfg["seed"], cfg["target_n"], verbose=True)
base, _ = P.finalize(ctx, S, cfg["target_n"], cfg["seed"])
for tag, syn in [("P1_seenfit_tilt_wide", P.finalize(ctx, dict(S, tilt_plan=plan_t), cfg["target_n"], cfg["seed"])[0]),
                 ("P2_seenfit_greedy_neutralise", apply_ops(base, ctx["spec"], plan_g))]:
    P.validate(syn, n, feats, cats, cfg["target_n"]); P.save_submission(syn, out / f"submission_UNSW-NB15_{tag}.csv")
    print("saved", tag, flush=True)
del ctx

# ---------- P3: classifier pushed to the extreme
S3 = dict(S, prefilter_frac=0.25, gamma=512, keep_frac=0.05)
ctx = P.prepare("vae", n, a, feats, nums, cats, name, S3, cfg["seed"], cfg["target_n"], verbose=True)
syn, info = P.finalize(ctx, S3, cfg["target_n"], cfg["seed"])
P.validate(syn, n, feats, cats, cfg["target_n"]); P.save_submission(syn, out / "submission_UNSW-NB15_P3_seenfit_hardtrim.csv")
log["P3_info"] = {k: info[k] for k in ["prefiltered_removed", "pool", "kept", "fallback_rows"]}
print("saved P3", log["P3_info"], flush=True)
json.dump(log, open(out / "overfit_probe_plans.json", "w"), indent=1, default=str)
print("ALL DONE", flush=True)
