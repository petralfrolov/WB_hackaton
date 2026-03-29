import numpy as np
import pandas as pd
import pytest

from optimizer_horizons import build_plan, optimize_horizon


def test_optimize_horizon_covers_and_respects_limits():
    capacities = np.array([5.0, 10.0])
    costs = np.array([50.0, 120.0])
    limits = np.array([None, 1])

    result = optimize_horizon(demand=12, capacities=capacities, costs=costs,
                              under_penalty=0.0, limits=limits)

    assert result[1] <= 1  # limit respected
    assert np.dot(capacities, result) >= 12  # demand covered
    assert (result >= 0).all()


def test_optimize_horizon_prefers_cheaper_per_unit_when_no_penalty():
    capacities = np.array([5.0, 10.0])
    costs = np.array([50.0, 300.0])  # van cheaper per unit

    result = optimize_horizon(demand=18, capacities=capacities, costs=costs,
                              under_penalty=0.0, limits=None)

    # Should choose only the cheap option (5-unit vans)
    assert result[1] == 0
    assert np.dot(capacities, result) >= 18


def test_optimize_horizon_zero_demand_returns_zeros():
    capacities = np.array([1.0, 10.0])
    costs = np.array([10.0, 100.0])

    result = optimize_horizon(demand=0, capacities=capacities, costs=costs,
                              under_penalty=10.0, limits=None)

    assert result.tolist() == [0, 0]


def test_build_plan_has_all_horizons():
    preds = {"pred_0_2h": 5.2, "pred_2_4h": 3.1, "pred_4_6h": 0.4}
    vehicles_cfg = {
        "vehicles": [
            {"vehicle_type": "van", "capacity_units": 5, "cost_per_km": 2.0, "available": 10},
            {"vehicle_type": "truck", "capacity_units": 10, "cost_per_km": 3.0, "available": 10},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 2,
        "wait_penalty_per_minute": 0.1,
        "underload_penalty_per_unit": 0.0,
    }

    plan = build_plan(route_id="100", timestamp="2025-05-01 10:00:00", preds=preds, vehicles_cfg=vehicles_cfg)

    horizons = set(plan["horizon"].unique())
    assert horizons == {"A: now", "B: +2h", "C: +4h", "D: +6h"}
    # Demand is rounded up inside build_plan
    assert (plan["demand"] >= 1).all()
    assert (plan["covered"] >= plan["demand"]).all()


def test_build_plan_wait_and_underload_costs():
    preds = {"pred_0_2h": 6.0, "pred_2_4h": 1.0, "pred_4_6h": 0.1}
    vehicles_cfg = {
        "vehicles": [
            {"vehicle_type": "solo", "capacity_units": 5, "cost_per_km": 1.0, "available": 10},
        ],
        "route_distance_km": 10,
        "initial_stock_units": 4,
        "wait_penalty_per_minute": 0.5,
        "underload_penalty_per_unit": 2.0,
    }

    plan = build_plan(route_id="200", timestamp="2025-05-01 10:00:00", preds=preds, vehicles_cfg=vehicles_cfg)

    horizon_b = plan[plan["horizon"] == "B: +2h"].iloc[0]

    expected_wait = 4 * 120 * 0.5  # init stock waits 120 minutes
    expected_fixed = 2 * (1.0 * 10)  # two vehicles * (cost_per_km * distance)
    expected_under = (10 - 6) * 2.0  # covered - demand

    assert horizon_b["cost_wait"] == pytest.approx(expected_wait)
    assert horizon_b["cost_underload"] == pytest.approx(expected_under)
    assert horizon_b["cost_total"] == pytest.approx(expected_wait + expected_fixed + expected_under)
