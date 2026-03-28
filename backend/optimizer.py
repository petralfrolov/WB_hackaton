"""Simple transport planner that converts forecast -> vehicle dispatch plan.

Usage (from repo root or backend/):
    python backend/optimizer.py --forecast submission_team_ready_models.csv \
                                 --out dispatch_plan_simple.csv

Inputs
------
- forecast CSV with columns: id, y_pred (submission format from the model).
- test_team_track.parquet to map id -> route_id, timestamp.
- train_team_track.parquet to map route_id -> office_from_id.
- vehicles.json with fields: vehicle_type, capacity_units, cost_per_trip.

Output
------
- dispatch_plan CSV with columns: office_from_id, route_id, dispatch_time,
  vehicle_type, vehicles_count.

Strategy implemented here (select via --mode)
--------------------------------------------
- single (по умолчанию): один самый выгодный по цене/юниту тип ТС, округление вверх.
- mincost: минимизируем общую стоимость рейсов на слот при покрытии спроса
  (целочисленный подбор, эквивалент простому ILP min Σ cost * x при Σ cap * x ≥ demand).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict

import pandas as pd


def load_forecast_submission(path: Path) -> pd.DataFrame:
    """Load submission CSV (id, y_pred)."""
    df = pd.read_csv(path)
    if not {"id", "y_pred"}.issubset(df.columns):
        raise ValueError("Forecast file must have columns [id, y_pred]")
    return df


def attach_route_time(forecast: pd.DataFrame, test_path: Path) -> pd.DataFrame:
    """Join route_id and timestamp from test parquet using id."""
    test_df = pd.read_parquet(test_path)
    if not {"id", "route_id", "timestamp"}.issubset(test_df.columns):
        raise ValueError("Test parquet must have columns id, route_id, timestamp")
    merged = forecast.merge(test_df[["id", "route_id", "timestamp"]], on="id", how="left")
    if merged[["route_id", "timestamp"]].isna().any().any():
        raise ValueError("Could not map some ids to route/timestamp; check test file")
    return merged[["route_id", "timestamp", "y_pred"]]


def attach_office(forecast: pd.DataFrame, train_path: Path) -> pd.DataFrame:
    """Add office_from_id via route_id mapping from train parquet."""
    train_df = pd.read_parquet(train_path, columns=["route_id", "office_from_id"])
    route_office = train_df.drop_duplicates("route_id")
    merged = forecast.merge(route_office, on="route_id", how="left")
    if merged["office_from_id"].isna().any():
        raise ValueError("Some route_id have no office_from_id mapping; check train file")
    return merged[["office_from_id", "route_id", "timestamp", "y_pred"]]


def load_vehicles(path: Path) -> List[Dict]:
    """Load vehicle config and compute cost per capacity unit."""
    vehicles = json.loads(Path(path).read_text())
    for v in vehicles:
        cap = float(v["capacity_units"])
        cost = float(v["cost_per_trip"])
        v["cost_per_unit"] = cost / cap if cap else float("inf")
    return vehicles


def build_plan_simple(forecast: pd.DataFrame, vehicles: List[Dict]) -> pd.DataFrame:
    """Plan with a single cheapest vehicle type per slot."""
    if not vehicles:
        raise ValueError("Vehicle list is empty")
    best = min(vehicles, key=lambda v: v["cost_per_unit"])

    rows = []
    for _, row in forecast.iterrows():
        demand = math.ceil(row["y_pred"])
        if demand <= 0:
            continue  # nothing to dispatch
        cnt = math.ceil(demand / best["capacity_units"])
        rows.append({
            "office_from_id": row.get("office_from_id"),
            "route_id": row["route_id"],
            "dispatch_time": row["timestamp"],
            "vehicle_type": best["vehicle_type"],
            "vehicles_count": cnt,
            "demand": demand,
        })
    return pd.DataFrame(rows)


def build_plan_mincost(forecast: pd.DataFrame, vehicles: List[Dict]) -> pd.DataFrame:
    """Minimize total trip cost per slot using integer DP (unbounded knapsack).

    For each (route, time) independently: find counts x_v (integers >=0) s.t.
    sum(capacity_v * x_v) >= demand and total cost is minimal.
    """
    if not vehicles:
        raise ValueError("Vehicle list is empty")

    # Precompute for speed
    caps = [int(v["capacity_units"]) for v in vehicles]
    costs = [float(v["cost_per_trip"]) for v in vehicles]
    names = [v["vehicle_type"] for v in vehicles]
    max_cap = max(caps)

    rows = []
    for _, row in forecast.iterrows():
        demand = math.ceil(row["y_pred"])
        if demand <= 0:
            continue

        target = demand + max_cap  # search slightly above demand for feasible cover
        INF = 1e18
        dp = [INF] * (target + 1)
        choice = [-1] * (target + 1)
        dp[0] = 0

        for v_idx, (cap, cost) in enumerate(zip(caps, costs)):
            for vol in range(cap, target + 1):
                prev = vol - cap
                cand = dp[prev] + cost
                if cand < dp[vol]:
                    dp[vol] = cand
                    choice[vol] = v_idx

        # pick best volume >= demand
        best_vol = None
        best_cost = INF
        for vol in range(demand, target + 1):
            if dp[vol] < best_cost:
                best_cost = dp[vol]
                best_vol = vol

        if best_vol is None or best_cost >= INF:
            raise RuntimeError("Could not find feasible plan for demand={}".format(demand))

        # backtrack counts
        counts = [0] * len(vehicles)
        vol = best_vol
        while vol > 0:
            v_idx = choice[vol]
            if v_idx == -1:
                break
            counts[v_idx] += 1
            vol -= caps[v_idx]

        covered = sum(c * cap for c, cap in zip(counts, caps))
        if covered < demand:
            # Fallback: simple greedy by cheapest cost/unit
            best_idx = min(range(len(vehicles)), key=lambda i: costs[i] / caps[i])
            extra = math.ceil((demand - covered) / caps[best_idx])
            counts[best_idx] += extra
            covered = sum(c * cap for c, cap in zip(counts, caps))

        for v_idx, cnt in enumerate(counts):
            if cnt == 0:
                continue
            rows.append({
                "office_from_id": row.get("office_from_id"),
                "route_id": row["route_id"],
                "dispatch_time": row["timestamp"],
                "vehicle_type": names[v_idx],
                "vehicles_count": cnt,
                "demand": demand,
            })

    return pd.DataFrame(rows)


def validate_coverage(plan: pd.DataFrame, vehicles: List[Dict]):
    """Check that for every slot sum(capacity*count) >= demand."""
    if plan.empty:
        return
    cap_map = {v["vehicle_type"]: v["capacity_units"] for v in vehicles}
    grp_cols = ["office_from_id", "route_id", "dispatch_time"]
    plan["covered"] = plan.apply(lambda r: cap_map[r["vehicle_type"]] * r["vehicles_count"], axis=1)
    agg = plan.groupby(grp_cols).agg(covered=("covered", "sum"), demand=("demand", "first"))
    bad = agg[agg["covered"] < agg["demand"]]
    if not bad.empty:
        raise RuntimeError(f"Coverage check failed for {len(bad)} slots; example:\n{bad.head()}")


def main():
    parser = argparse.ArgumentParser(description="Transport planner: forecast -> dispatch plan")
    parser.add_argument("--forecast", required=True, type=Path, help="Path to forecast CSV (id,y_pred)")
    parser.add_argument("--vehicles", default=Path("vehicles.json"), type=Path, help="Path to vehicles.json")
    parser.add_argument("--test-path", default=Path("test_team_track.parquet"), type=Path,
                        help="Path to test parquet (needed to map id -> route_id, timestamp)")
    parser.add_argument("--train-path", default=Path("train_team_track.parquet"), type=Path,
                        help="Path to train parquet (to map route_id -> office_from_id)")
    parser.add_argument("--out", default=Path("dispatch_plan_simple.csv"), type=Path, help="Where to save the plan CSV")
    parser.add_argument("--mode", choices=["single", "mincost"], default="single",
                        help="single = one cheapest type; mincost = minimal cost cover via DP")
    args = parser.parse_args()

    forecast = load_forecast_submission(args.forecast)
    forecast = attach_route_time(forecast, args.test_path)
    forecast = attach_office(forecast, args.train_path)
    vehicles = load_vehicles(args.vehicles)

    if args.mode == "single":
        plan = build_plan_simple(forecast, vehicles)
    else:
        plan = build_plan_mincost(forecast, vehicles)

    validate_coverage(plan, vehicles)

    plan.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  rows={len(plan)}  vehicle_type={plan['vehicle_type'].unique() if not plan.empty else 'n/a'}")


if __name__ == "__main__":
    main()
