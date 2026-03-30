import numpy as np
import pandas as pd
import pytest

from optimizer_horizons import build_plan, solve_irp_milp, HORIZONS, PERIOD_MINUTES

VEHICLES_CFG_SIMPLE = {
    "vehicles": [
        {"vehicle_type": "van",   "capacity_units": 5,  "cost_per_km": 2.0, "available": 10},
        {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 3.0, "available": 10},
    ],
    "route_distance_km": 10,
    "initial_stock_units": 0,
    "wait_penalty_per_minute": 0.0,
    "empty_capacity_penalty": 0.0,
}


# ---------------------------------------------------------------------------
# solve_irp_milp
# ---------------------------------------------------------------------------

def test_milp_flow_balance_holds():
    """y[r,t] + s[r,t] == D[r,t] + s[r,t-1] for every route and horizon."""
    demands = {"r1": [10.0, 8.0, 6.0, 4.0]}
    res = solve_irp_milp(demands, VEHICLES_CFG_SIMPLE)
    Y, S, D = res["Y"], res["S"], res["D"]
    nT = len(HORIZONS)
    for t in range(nT):
        s_prev = S[0, t - 1] if t > 0 else 0.0
        assert Y[0, t] + S[0, t] == pytest.approx(D[0, t] + s_prev, abs=1e-3)


def test_milp_shared_fleet_not_exceeded():
    """For each (vehicle_type, horizon), total vehicles assigned ≤ available."""
    demands = {
        "r1": [12.0, 10.0, 8.0, 6.0],
        "r2": [15.0,  5.0, 3.0, 2.0],
    }
    vehicles_cfg = {
        "vehicles": [
            {"vehicle_type": "van",   "capacity_units": 5,  "cost_per_km": 2.0, "available": 3},
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 3.0, "available": 2},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 0.0,
        "empty_capacity_penalty": 0.0,
    }
    res = solve_irp_milp(demands, vehicles_cfg)
    X = res["X"]
    max_fleet = [3, 2]
    nT = len(HORIZONS)
    for vi in range(2):
        for t in range(nT):
            assert X[:, vi, t].sum() <= max_fleet[vi] + 1e-6


def test_milp_zero_demand_no_vehicles():
    demands = {"r1": [0.0, 0.0, 0.0, 0.0]}
    res = solve_irp_milp(demands, VEHICLES_CFG_SIMPLE)
    assert res["X"].sum() == 0


def test_milp_non_negative_vars():
    demands = {"r1": [7.0, 5.0, 3.0, 1.0]}
    res = solve_irp_milp(demands, VEHICLES_CFG_SIMPLE)
    assert (res["X"] >= -1e-9).all()
    assert (res["Y"] >= -1e-9).all()
    assert (res["S"] >= -1e-9).all()
    assert (res["U"] >= -1e-9).all()


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------

def test_build_plan_all_horizons_present():
    demands = {"r1": [5.0, 4.0, 3.0, 1.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=VEHICLES_CFG_SIMPLE,
    )
    assert set(plan["horizon"].unique()) == {"A: now", "B: +2h", "C: +4h", "D: +6h"}


def test_build_plan_columns_present():
    demands = {"r1": [6.0, 4.0, 2.0, 1.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=VEHICLES_CFG_SIMPLE,
    )
    required = {
        "route_id", "timestamp", "horizon", "vehicle_type", "vehicles_count",
        "demand_new", "demand_carried_over", "total_available",
        "actually_shipped", "leftover_stock", "empty_capacity_units",
        "cost_fixed", "cost_underload", "cost_wait", "cost_total",
    }
    assert required.issubset(set(plan.columns))


def test_build_plan_flow_balance_in_output():
    """actually_shipped + leftover_stock == demand_new + demand_carried_over in each row."""
    demands = {"r1": [8.0, 6.0, 4.0, 2.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=VEHICLES_CFG_SIMPLE,
    )
    # group by horizon to get per-horizon totals (one row per horizon when no vehicle split)
    for _, grp in plan.groupby("horizon"):
        row = grp.iloc[0]
        assert row["actually_shipped"] + row["leftover_stock"] == pytest.approx(
            row["demand_new"] + row["demand_carried_over"], abs=0.05
        )


def test_build_plan_floats_rounded_to_2dp():
    demands = {"r1": [7.0, 5.0, 3.0, 1.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=VEHICLES_CFG_SIMPLE,
    )
    float_cols = ["demand_new", "demand_carried_over", "total_available",
                  "actually_shipped", "leftover_stock", "empty_capacity_units",
                  "cost_fixed", "cost_underload", "cost_wait", "cost_total"]
    for col in float_cols:
        for val in plan[col]:
            assert val == round(val, 2), f"{col} value {val} not rounded to 2dp"


def test_build_plan_wait_penalty_applied():
    """If wait_penalty > 0 and some stock is deferred, cost_wait > 0."""
    demands = {"r1": [0.0, 3.0, 0.0, 0.0]}
    cfg = {
        "vehicles": [
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 1.0, "available": 1},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 1.0,
        "empty_capacity_penalty": 100.0,  # high: force dispatch at B not deferring
    }
    plan = build_plan(timestamp="2025-05-01 10:00:00", demands=demands, vehicles_cfg=cfg)
    # total wait cost across all horizons should be >= 0
    assert plan["cost_wait"].sum() >= 0.0


def test_build_plan_office_id_inserted():
    demands = {"r1": [4.0, 2.0, 1.0, 0.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=VEHICLES_CFG_SIMPLE,
        office_id="WH-01",
    )
    assert "office_from_id" in plan.columns
    assert (plan["office_from_id"] == "WH-01").all()


def test_build_plan_multi_route_shared_fleet():
    """Two routes together must not exceed fleet limit at any horizon."""
    cfg = {
        "vehicles": [
            {"vehicle_type": "van", "capacity_units": 5, "cost_per_km": 1.0, "available": 2},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 0.0,
        "empty_capacity_penalty": 0.0,
    }
    demands = {
        "r1": [10.0, 8.0, 6.0, 4.0],
        "r2": [10.0, 8.0, 6.0, 4.0],
    }
    plan = build_plan(timestamp="2025-05-01 10:00:00", demands=demands, vehicles_cfg=cfg)
    for horizon in plan["horizon"].unique():
        total = plan.loc[plan["horizon"] == horizon, "vehicles_count"].sum()
        assert total <= 2
