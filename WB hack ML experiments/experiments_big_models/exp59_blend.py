"""Blend: 0.6 * exp51 + 0.4 * exp58c_cal51"""
import sys
import pandas as pd

sub51  = pd.read_csv("submission_team_exp51_hyperparam_diversity.csv")
sub58c = pd.read_csv("submission_team_exp58c_cal51.csv")

target_cols = [c for c in sub51.columns if c.startswith("target_")]

blend = sub51.copy()
blend[target_cols] = 0.6 * sub51[target_cols].values + 0.4 * sub58c[target_cols].values

out = "submission_team_exp59_blend51_58c.csv"
blend.to_csv(out, index=False)
print(f"Saved: {out}  ({len(blend)} rows)")
print("Blend: 0.6 × exp51 + 0.4 × exp58c_cal51")
