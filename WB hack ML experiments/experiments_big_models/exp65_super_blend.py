"""Experiment 65: Super-ensemble blend of best existing submissions.

Zero-compute experiment: average predictions from the top local-val submissions.
This is a "poor man's stacking" — combining independently trained models that
were never explicitly ensembled together.

Submissions to blend (all available, sorted by local val score):
  exp57  (0.239609) — leaves=511+255, cs=0.5
  exp53  (0.239683) — pure GroupB cs=0.5
  exp56  (0.239683) — no-anchor cs=0.5/0.7/0.35
  exp52  (0.240134) — feature+reg diversity
  exp51  (0.240207) — hyperparam diversity (public LB=0.2502 — BEST public)
  + exp61, exp62, exp63, exp64 if available

Weighting: uses inverse-loss weighting (lower local val → higher weight).
Also saves an equal-weight variant for comparison.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

EXPERIMENT_NAME = "exp65_super_blend"

# Submissions directory
SUB_DIR = Path("experiments_big_models")
ROOT_DIR = Path(".")

# Priority submissions: (filename_stem, local_val_total)
# Lower total = better = higher weight
BLEND_CANDIDATES = [
    ("submission_team_exp57_capacity_diversity",  0.239609),
    ("submission_team_exp53_pure_groupb",          0.239683),
    ("submission_team_exp56_no_anchor",            0.239683),
    ("submission_team_exp52_feature_reg_diversity",0.240134),
    ("submission_team_exp51_hyperparam_diversity", 0.240207),
]

# Also try to include exp61-64 if they exist
OPTIONAL_CANDIDATES = [
    "submission_team_exp61_roll_ratio_features",
    "submission_team_exp62_bynode_diversity",
    "submission_team_exp63_perstep_alpha",
    "submission_team_exp64_halfhour_of_week",
]

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("  Zero-compute blend of best submissions")
print("=" * 60)
sys.stdout.flush()


def load_sub(stem):
    """Try to load submission from multiple locations."""
    for base in [SUB_DIR, ROOT_DIR]:
        p = base / f"{stem}.csv"
        if p.exists():
            return pd.read_csv(p), str(p)
    return None, None


# Load mandatory candidates
subs = []
weights = []
loaded_names = []

for stem, local_val in BLEND_CANDIDATES:
    df, path = load_sub(stem)
    if df is None:
        print(f"  SKIP (not found): {stem}")
        continue
    # weight = 1 / local_val (better score → higher weight)
    w = 1.0 / local_val
    subs.append(df)
    weights.append(w)
    loaded_names.append(stem)
    print(f"  Loaded: {stem}  (local_val={local_val:.6f}  weight={w:.4f})  [{path}]")
    sys.stdout.flush()

# Load optional candidates (get their local_val from experiments.json if available)
exp_path = Path("experiments.json")
exps_data = json.loads(exp_path.read_text(encoding="utf-8"))
exp_map = {e["name"]: e.get("total", 0.25) for e in exps_data}

for stem in OPTIONAL_CANDIDATES:
    df, path = load_sub(stem)
    if df is None:
        print(f"  OPTIONAL SKIP: {stem}")
        continue
    # Derive experiment name from stem
    exp_name = stem.replace("submission_team_", "")
    local_val = exp_map.get(exp_name, 0.240)
    w = 1.0 / local_val
    subs.append(df)
    weights.append(w)
    loaded_names.append(stem)
    print(f"  Optional: {stem}  (local_val={local_val:.6f}  weight={w:.4f})  [{path}]")
    sys.stdout.flush()

if not subs:
    print("ERROR: No submissions loaded!")
    sys.exit(1)

print(f"\nTotal submissions to blend: {len(subs)}")
sys.stdout.flush()

target_cols = [c for c in subs[0].columns if c != "id"]
id_col = subs[0]["id"]

# Normalize weights
weights = np.array(weights)
weights_norm = weights / weights.sum()
print(f"Normalized weights: {[round(w, 4) for w in weights_norm]}")
sys.stdout.flush()

# Inverse-loss weighted blend
blend_inv = subs[0].copy()
blend_inv[target_cols] = sum(
    w * df[target_cols].values for w, df in zip(weights_norm, subs)
)

# Equal-weight blend
blend_eq = subs[0].copy()
blend_eq[target_cols] = sum(df[target_cols].values for df in subs) / len(subs)

# Save both
out_inv = f"submission_team_{EXPERIMENT_NAME}_invloss.csv"
out_eq  = f"submission_team_{EXPERIMENT_NAME}_equal.csv"
blend_inv.to_csv(out_inv, index=False)
blend_eq.to_csv(out_eq, index=False)
print(f"Saved: {out_inv}  ({len(blend_inv)} rows)")
print(f"Saved: {out_eq}   ({len(blend_eq)} rows)")
sys.stdout.flush()

# Best single sub as reference
best_stem, best_local = BLEND_CANDIDATES[0]
best_df, _ = load_sub(best_stem)
if best_df is not None:
    diff_inv = (blend_inv[target_cols].values - best_df[target_cols].values)
    diff_eq  = (blend_eq[target_cols].values  - best_df[target_cols].values)
    print(f"\nvs best single ({best_stem}):")
    print(f"  Inv-weighted MAD from best: {np.abs(diff_inv).mean():.4f}")
    print(f"  Equal-weight MAD from best: {np.abs(diff_eq).mean():.4f}")
    sys.stdout.flush()

# Update experiments.json
exps_data.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "blended": loaded_names,
        "weights": [round(float(w), 6) for w in weights_norm],
        "n_subs": len(subs),
        "method": "inverse_loss_weighted + equal_weight",
    },
    "wape": None,
    "rbias": None,
    "total": None,
    "note": (
        f"Super-blend of {len(subs)} submissions: {loaded_names}. "
        f"Inv-loss weights: {[round(float(w),4) for w in weights_norm]}. "
        f"Saved both inv-weighted and equal-weight variants. "
        f"No local val (post-hoc blend — submit both to see public LB)."
    ),
})
exp_path.write_text(json.dumps(exps_data, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Blended {len(subs)} submissions")
print(f"  Inv-weighted blend → {out_inv}")
print(f"  Equal-weight blend → {out_eq}")
print(f"  Submit both variants to public LB!")
print("=" * 60)
