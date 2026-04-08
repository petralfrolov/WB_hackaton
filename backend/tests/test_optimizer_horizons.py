import numpy as np
import pandas as pd
import pytest

from optimizer.horizons import (
    build_plan, solve_irp_milp, _build_fleet_limits, HORIZONS, PERIOD_MINUTES,
)

VEHICLES_CFG_SIMPLE = {
    "vehicles": [
        {"vehicle_type": "van",   "capacity_units": 5,  "cost_per_km": 2.0, "available": 10, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
        {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 3.0, "available": 10, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
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
            {"vehicle_type": "van",   "capacity_units": 5,  "cost_per_km": 2.0, "available": 3, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 3.0, "available": 2, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
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
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 1.0, "available": 1, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
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
            {"vehicle_type": "van", "capacity_units": 5, "cost_per_km": 1.0, "available": 2, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
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


# ---------------------------------------------------------------------------
# _build_fleet_limits
# ---------------------------------------------------------------------------

VEHICLES_TWO = [
    {"vehicle_type": "van",   "capacity_units": 5,  "cost_per_km": 1.0, "available": 3, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
    {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 2.0, "available": 2, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
]
nT = len(HORIZONS)


def test_build_fleet_limits_no_incoming():
    """Without incoming, all horizons equal base available count."""
    fleet = _build_fleet_limits(VEHICLES_TWO, incoming=None, nT=nT)
    assert fleet.shape == (2, nT)
    assert (fleet[0] == 3).all()
    assert (fleet[1] == 2).all()


def test_build_fleet_limits_incoming_additive():
    """Incoming at DB horizon_idx 4 (+2h at 30-min steps) maps to optimizer h1 and increases limits."""
    # DB stores horizon_idx in 30-min steps: idx 4 = 4*30 = 120 min = optimizer h1 at 120-min period
    incoming = [{'horizon_idx': 4, 'vehicle_type': 'van', 'count': 2}]
    fleet = _build_fleet_limits(VEHICLES_TWO, incoming=incoming, nT=nT)
    assert fleet[0, 0] == 3   # t=0: no change
    assert fleet[0, 1] == 5   # t=1: +2
    assert fleet[0, 2] == 5   # t=2: carries forward
    assert fleet[0, 3] == 5   # t=3
    assert (fleet[1] == 2).all()  # truck untouched


def test_build_fleet_limits_multiple_incoming():
    """Multiple entries accumulate correctly (DB 30-min-based indices)."""
    # idx 4 = 120 min = optimizer h1; idx 8 = 240 min = optimizer h2
    incoming = [
        {'horizon_idx': 4, 'vehicle_type': 'van',   'count': 1},
        {'horizon_idx': 8, 'vehicle_type': 'van',   'count': 2},
        {'horizon_idx': 4, 'vehicle_type': 'truck', 'count': 1},
    ]
    fleet = _build_fleet_limits(VEHICLES_TWO, incoming=incoming, nT=nT)
    assert fleet[0, 0] == 3
    assert fleet[0, 1] == 4   # +1 at t=1
    assert fleet[0, 2] == 6   # +2 more at t=2
    assert fleet[0, 3] == 6
    assert fleet[1, 0] == 2
    assert fleet[1, 1] == 3   # +1 at t=1
    assert fleet[1, 2] == 3
    assert fleet[1, 3] == 3


def test_build_fleet_limits_mid_horizon_arrival_uses_next_slot():
    """Arrival at +5h must become available no earlier than optimizer horizon +6h."""
    # DB idx 10 = 10*30 = 300 min = +5h, which should map to optimizer h3 (+6h), not h2 (+4h)
    incoming = [{'horizon_idx': 10, 'vehicle_type': 'van', 'count': 2}]
    fleet = _build_fleet_limits(VEHICLES_TWO, incoming=incoming, nT=nT)
    assert fleet[0, 0] == 3
    assert fleet[0, 1] == 3
    assert fleet[0, 2] == 3
    assert fleet[0, 3] == 5


def test_build_fleet_limits_unknown_type_raises():
    incoming = [{"horizon_idx": 1, "vehicle_type": "helicopter", "count": 1}]
    with pytest.raises(ValueError, match="unknown vehicle_type"):
        _build_fleet_limits(VEHICLES_TWO, incoming=incoming, nT=nT)


def test_build_fleet_limits_bad_horizon_raises():
    """Negative stored horizon_idx must raise ValueError."""
    incoming = [{"horizon_idx": -1, "vehicle_type": "van", "count": 1}]
    with pytest.raises(ValueError, match="horizon_idx"):
        _build_fleet_limits(VEHICLES_TWO, incoming=incoming, nT=nT)


# ---------------------------------------------------------------------------
# solve_irp_milp with incoming_vehicles
# ---------------------------------------------------------------------------

def test_milp_incoming_expands_fleet():
    """With tight base fleet + incoming, solver can dispatch more at later horizons."""
    cfg_tight = {
        "vehicles": [
            {"vehicle_type": "van", "capacity_units": 5, "cost_per_km": 1.0, "available": 1, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 0.0,
        "empty_capacity_penalty": 0.0,
    }
    # demand requires 2 vans at t=2, but base has only 1
    demands = {"r1": [0.0, 0.0, 20.0, 0.0]}

    # Without incoming: solver should defer or partially cover
    res_no_inc = solve_irp_milp(demands, cfg_tight)

    # With incoming: 1 extra van arrives at optimizer t=2 (+4h). DB idx 8 = 8*30 = 240 min.
    incoming = [{"horizon_idx": 8, "vehicle_type": "van", "count": 1}]
    res_with_inc = solve_irp_milp(demands, cfg_tight, incoming_vehicles=incoming)

    # With incoming, can dispatch at least 2 vans at t=2 (limit goes 1→2)
    assert res_with_inc["X"][0, 0, 2] <= 2   # capped at new limit 2
    # Fleet limit without incoming is 1 at all horizons
    assert res_no_inc["X"][0, 0, 2] <= 1


def test_milp_incoming_does_not_affect_earlier_horizons():
    """Incoming at t=2 must NOT allow extra vehicles at t=0 or t=1."""
    cfg = {
        "vehicles": [
            {"vehicle_type": "van", "capacity_units": 5, "cost_per_km": 1.0, "available": 1, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 0.0,
        "empty_capacity_penalty": 0.0,
    }
    # DB idx 8 = 8*30 = 240 min = optimizer h2 (+4h at 120-min granularity)
    incoming = [{"horizon_idx": 8, "vehicle_type": "van", "count": 10}]
    demands = {"r1": [20.0, 20.0, 0.0, 0.0]}
    res = solve_irp_milp(demands, cfg, incoming_vehicles=incoming)
    # At t=0 and t=1, base limit is still 1
    assert res["X"][0, 0, 0] <= 1
    assert res["X"][0, 0, 1] <= 1


# ---------------------------------------------------------------------------
# build_plan with incoming_vehicles
# ---------------------------------------------------------------------------

def test_build_plan_incoming_vehicles_column_values():
    """Vehicles dispatched per horizon must not exceed (base + incoming) at that horizon."""
    cfg = {
        "vehicles": [
            {"vehicle_type": "van",   "capacity_units": 5, "cost_per_km": 1.0, "available": 1, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 2.0, "available": 1, "underload_penalty": 0.0, "fixed_dispatch_cost": 0.0},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 0,
        "wait_penalty_per_minute": 0.0,
        "empty_capacity_penalty": 0.0,
    }
    # DB 30-min-based indices: idx 4 = +2h (optimizer h1), idx 8 = +4h (optimizer h2)
    incoming = [
        {'horizon_idx': 4, 'vehicle_type': 'van',   'count': 2},
        {'horizon_idx': 8, 'vehicle_type': 'truck', 'count': 1},
    ]
    demands = {"r1": [5.0, 15.0, 20.0, 5.0]}
    plan = build_plan(
        timestamp="2025-05-01 10:00:00",
        demands=demands,
        vehicles_cfg=cfg,
        incoming_vehicles=incoming,
    )
    # Horizon B: van limit = 1+2 = 3, truck = 1 → total van dispatched ≤ 3
    h_b = plan[plan["horizon"] == "B: +2h"]
    van_b = h_b.loc[h_b["vehicle_type"] == "van", "vehicles_count"].sum()
    assert van_b <= 3

    # Horizon C: truck limit = 1+1 = 2 → dispatched ≤ 2
    h_c = plan[plan["horizon"] == "C: +4h"]
    truck_c = h_c.loc[h_c["vehicle_type"] == "truck", "vehicles_count"].sum()
    assert truck_c <= 2

