"""New optimizer: uses ml_prediction horizons (0-2h, 2-4h, 4-6h) per route.

Workflow:
- Получаем прогноз по route_id и timestamp через функции из ml_prediction.py.
- Для каждого горизонта решаем задачу min Σ cost_i * x_i при ограничении
  Σ capacity_i * x_i >= demand. Решаем SLSQP (scipy), потом округляем вверх.
- На выход: CSV с планом по каждому горизонту и типу ТС.
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml_prediction import (
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_PATH,
    load_models,
    prepare_feature_matrix,
    predict_for_route_timestamp,
)


def load_route_office_map(train_path: Path) -> Dict[str, str]:
    df = pd.read_parquet(train_path, columns=["route_id", "office_from_id"])
    df = df.drop_duplicates("route_id")
    return {str(r): str(o) for r, o in zip(df["route_id"], df["office_from_id"])}


def optimize_horizon(demand: float, capacities: np.ndarray, costs: np.ndarray, under_penalty: float,
                     limits: np.ndarray | None = None) -> np.ndarray:
    """SLSQP continuous solve -> ceil to integers, ensure coverage; учитывает лимиты available."""
    n = len(capacities)
    if demand <= 0:
        return np.zeros(n, dtype=int)

    # objective: cost · x + under_penalty * max(cap·x - demand, 0)
    def obj(x):
        cover = float(np.dot(capacities, x))
        empty = max(0.0, cover - demand)
        return float(np.dot(costs, x) + under_penalty * empty)

    # constraint: capacity · x >= demand
    cons = ({"type": "ineq", "fun": lambda x: float(np.dot(capacities, x) - demand)})
    bounds = []
    for i in range(n):
        ub = None if limits is None else limits[i]
        bounds.append((0, ub))
    x0 = np.full(n, demand / (capacities.max() * n))

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    x = res.x if res.success else x0
    x_int = np.ceil(x).astype(int)

    covered = np.dot(capacities, x_int)
    if covered < demand:
        # добиваем самым выгодным по цене/юниту
        cost_per_unit = costs / capacities
        best = int(cost_per_unit.argmin())
        extra = math.ceil((demand - covered) / capacities[best])
        x_int[best] += extra
    return x_int


def build_plan(route_id: str, timestamp: str, preds: Dict[str, float], vehicles_cfg: Dict) -> pd.DataFrame:
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    caps = np.array([v["capacity_units"] for v in vehicles], dtype=float)
    distance_km = float(vehicles_cfg.get("route_distance_km", 15.0))
    costs = np.array([v.get("cost_per_km", 0) * distance_km for v in vehicles], dtype=float)
    names = [v["vehicle_type"] for v in vehicles]
    limits = np.array([v.get("available", np.inf) for v in vehicles], dtype=float)

    init_stock = float(vehicles_cfg.get("initial_stock_units", 0))
    wait_penalty = float(vehicles_cfg.get("wait_penalty_per_minute", 0))
    under_penalty = float(vehicles_cfg.get("underload_penalty_per_unit", 0))

    horizons = [
        ("A: now", init_stock, 0),
        ("B: +2h", preds["pred_0_2h"], 120),
        ("C: +4h", preds["pred_2_4h"], 240),
        ("D: +6h", preds["pred_4_6h"], 360),
    ]

    cohorts = [(init_stock, 0)]  # (size, arrival_minute)
    rows = []

    for label, demand, horizon_min in horizons:
        counts = optimize_horizon(demand, caps, costs, under_penalty, limits)
        covered = float(np.dot(counts, caps))
        # Штрафуем незаполненную вместимость: сколько места осталось пустым.
        under_units = max(0.0, covered - demand)
        under_cost = under_units * under_penalty

        wait_cost = 0.0
        if not label.startswith("A"):
            for size, arr_min in cohorts:
                wait_cost += size * max(horizon_min - arr_min, 0) * wait_penalty
            cohorts.append((demand, horizon_min))

        fixed_cost = float(np.dot(counts, costs))
        total_cost = fixed_cost + under_cost + wait_cost

        for name, cnt, cap, cost in zip(names, counts, caps, costs):
            if cnt <= 0:
                continue
            rows.append({
                "route_id": route_id,
                "timestamp": timestamp,
                "horizon": label,
                "vehicle_type": name,
                "vehicles_count": int(cnt),
                "demand": float(demand),
                "covered": float(cnt * cap),
                "cost_fixed": float(cnt * cost),
                "cost_underload": under_cost,
                "cost_wait": wait_cost,
                "cost_total": total_cost,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Optimize transport per 0-2/2-4/4-6h horizons")
    parser.add_argument("--route_id", required=True)
    parser.add_argument("--timestamp", required=True, help="Timestamp like 2025-05-30 10:30:00")
    parser.add_argument("--train_path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--vehicles", default="vehicles.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    route_id = str(args.route_id)
    ts = args.timestamp

    # load data
    vehicles_cfg = json.loads(Path(args.vehicles).read_text())
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    office_map = load_route_office_map(Path(args.train_path))
    office_id = office_map.get(route_id, "")

    models = load_models(Path(args.models_dir))
    X_all, feature_cols = prepare_feature_matrix(Path(args.train_path))

    preds = predict_for_route_timestamp(X_all, feature_cols, models, route_id=route_id, timestamp=ts)

    plan = build_plan(route_id, ts, preds, vehicles_cfg)
    if office_id:
        plan.insert(0, "office_from_id", office_id)

    # Выводим в консоль вместо сохранения CSV
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(plan.to_string(index=False))
    print(f"\nCoverage check: min covered-demand = {(plan['covered'] - plan['demand']).min():.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
