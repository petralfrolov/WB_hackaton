"""Experiment 68: super-blend of best submissions (fixed, no unicode).

Blends top submissions weighted by inverse local-val score.
Also includes exp66 and exp67 if already saved.
"""

import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

EXPERIMENT_NAME = "exp68_super_blend"

SUB_DIR  = Path("experiments_big_models")
ROOT_DIR = Path(".")

# (stem, local_val – lower is better)
PRIORITY = [
    ("submission_team_exp57_capacity_diversity",    0.239609),
    ("submission_team_exp53_pure_groupb",           0.239683),
    ("submission_team_exp56_no_anchor",             0.239683),
    ("submission_team_exp61_roll_ratio_features",   0.240000),  # approx, 0.2501 public
    ("submission_team_exp52_feature_reg_diversity", 0.240134),
    ("submission_team_exp51_hyperparam_diversity",  0.240207),  # best public 0.2502
]
OPTIONAL = [
    "submission_team_exp66_roll_ratio_struct_diversity",
    "submission_team_exp67_roll_ratio_cap_diversity",
]

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("  Super-blend of best submissions (inv-loss + equal variants)")
print("=" * 60)
sys.stdout.flush()


def load_sub(stem):
    for base in [SUB_DIR, ROOT_DIR]:
        p = base / f"{stem}.csv"
        if p.exists():
            return pd.read_csv(p), str(p)
    return None, None


exp_path = Path("experiments.json")
exp_map = {e["name"]: e.get("total", 0.25)
           for e in json.loads(exp_path.read_text(encoding="utf-8"))}

subs, weights, names = [], [], []

for stem, lv in PRIORITY:
    df, path = load_sub(stem)
    if df is None:
        print(f"  SKIP: {stem}")
        continue
    w = 1.0 / lv
    subs.append(df); weights.append(w); names.append(stem)
    print(f"  OK  : {stem}  lv={lv:.6f}  w={w:.4f}  [{path}]")
    sys.stdout.flush()

for stem in OPTIONAL:
    df, path = load_sub(stem)
    if df is None:
        print(f"  OPT SKIP: {stem}")
        continue
    exp_name = stem.replace("submission_team_", "")
    lv = exp_map.get(exp_name, 0.240)
    w = 1.0 / lv
    subs.append(df); weights.append(w); names.append(stem)
    print(f"  OPT : {stem}  lv={lv:.6f}  w={w:.4f}  [{path}]")
    sys.stdout.flush()

assert subs, "No submissions loaded!"
print(f"\nTotal to blend: {len(subs)}")

target_cols = [c for c in subs[0].columns if c != "id"]
wn = np.array(weights) / sum(weights)
print(f"Normalized weights: {[round(float(x),4) for x in wn]}")
sys.stdout.flush()

blend_inv = subs[0].copy()
blend_inv[target_cols] = sum(w * df[target_cols].values for w, df in zip(wn, subs))

blend_eq = subs[0].copy()
blend_eq[target_cols] = sum(df[target_cols].values for df in subs) / len(subs)

out_inv = f"submission_team_{EXPERIMENT_NAME}_invloss.csv"
out_eq  = f"submission_team_{EXPERIMENT_NAME}_equal.csv"
blend_inv.to_csv(out_inv, index=False)
blend_eq.to_csv(out_eq, index=False)
print(f"Saved: {out_inv}  ({len(blend_inv)} rows)")
print(f"Saved: {out_eq}  ({len(blend_eq)} rows)")
sys.stdout.flush()

exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"blended": names, "weights": [round(float(x),6) for x in wn],
               "n_subs": len(subs), "method": "inv_loss + equal"},
    "wape": None, "rbias": None, "total": None,
    "note": (f"Super-blend of {len(subs)} subs. "
             f"Inv-loss weights: {[round(float(x),4) for x in wn]}. "
             f"Saved inv-weighted={out_inv} and equal={out_eq}.")})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Blended {len(subs)} submissions")
print(f"  inv-weighted: {out_inv}")
print(f"  equal-weight: {out_eq}")
print("=" * 60)
