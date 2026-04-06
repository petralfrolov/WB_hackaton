"""IRP MILP optimizer: joint dispatch planning across all routes and horizons.

Workflow:
- Получаем прогноз по route_id(s) и timestamp через ml_prediction.py.
- Для всех маршрутов офиса решаем единый MILP (scipy.optimize.milp):
      min  Σ_{r,v,t} Cost_{v,r}·x_{r,v,t}
         + Σ_{r,t}   P_empty·u_{r,t}
         + Σ_{r,t}   P_wait·s_{r,t}
      s.t.
        (1) Баланс потока  : y[r,t] + s[r,t] − s[r,t−1] = D[r,t]   ∀r,t
        (2) Вместимость ТС : Σ_v Cap_v·x[r,v,t] = y[r,t] + u[r,t]  ∀r,t
        (3) Общий автопарк : Σ_r x[r,v,t] ≤ MaxV_v[v,t]             ∀v,t
            (при --global_fleet: Σ_{r,τ≤t} x[r,v,τ] ≤ MaxV_v[v,t]  ∀v,t)
        (4) x ∈ Z≥0,  y,s,u ≥ 0
  Переменные:
        x[r,v,t]  — кол-во ТС типа v на маршруте r в горизонте t (целое)
        y[r,t]    — фактически отправленный объём (ед. товара)
        s[r,t]    — остаток/очередь, переходящий в t+1
        u[r,t]    — объём пустого пространства в кузовах (недогруз)
- На выход: CSV с планом, округлённым до 2 знаков.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from ml.prediction import (
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_PATH,
    load_models,
    prepare_feature_matrix,
    predict_for_route_timestamp,
)

# Горизонты планирования: (метка, смещение от «сейчас» в минутах)
HORIZONS: List[tuple[str, int]] = [
    ("A: now",   0),
    ("B: +2h", 120),
    ("C: +4h", 240),
    ("D: +6h", 360),
]
PERIOD_MINUTES = 120  # длительность одного горизонта, мин


def make_horizons(granularity: float = 2.0) -> tuple:
    """Build horizon list and period minutes for a given granularity.

    Args:
        granularity: Forecast granularity in hours (0.5, 1.0, or 2.0).

    Returns:
        Tuple of (horizons_list, period_minutes) where horizons_list is
        [(label, offset_minutes), ...] including the initial "A: now" slot.
    """
    if granularity == 2.0:
        return HORIZONS, 120
    period = int(granularity * 60)
    n_future = int(6.0 / granularity)
    labels_abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    horizons = [("A: now", 0)]
    for i in range(1, n_future + 1):
        offset = i * period
        h_val = offset / 60
        if h_val == int(h_val):
            label = f"{labels_abc[i]}: +{int(h_val)}h"
        else:
            label = f"{labels_abc[i]}: +{h_val}h"
        horizons.append((label, offset))
    return horizons, period


def load_route_office_map(train_path: Path) -> Dict[str, str]:
    """Read the parquet file and return a mapping of route_id to office_from_id.

    Args:
        train_path: Path to the training parquet file.

    Returns:
        Dict mapping ``str(route_id)`` → ``str(office_from_id)``.
    """
    df = pd.read_parquet(train_path, columns=["route_id", "office_from_id"])
    df = df.drop_duplicates("route_id")
    return {str(r): str(o) for r, o in zip(df["route_id"], df["office_from_id"])}


def _build_fleet_limits(
    vehicles: List[Dict],
    incoming: Optional[List[Dict]],
    nT: int,
) -> np.ndarray:
    """Вернуть матрицу max_fleet[v, t] — сколько ТС типа v доступно начиная с горизонта t.

    incoming — список записей {horizon_idx: int, vehicle_type: str, count: int}.
    Каждая запись увеличивает доступный парк начиная с horizon_idx и далее.
    """
    v_names = [v["vehicle_type"] for v in vehicles]
    base = np.array([float(v.get("available", 1000)) for v in vehicles])  # (nV,)
    fleet = np.tile(base[:, None], (1, nT))  # (nV, nT)

    for entry in (incoming or []):
        h = int(entry["horizon_idx"])
        vtype = entry["vehicle_type"]
        cnt = int(entry["count"])
        if vtype not in v_names:
            raise ValueError(f"incoming_vehicles: unknown vehicle_type '{vtype}'")
        vi = v_names.index(vtype)
        # Map original 4-horizon indices to current nT horizons
        if not (0 <= h < nT):
            # Skip entries beyond current horizon count
            if h >= nT:
                continue
            raise ValueError(f"incoming_vehicles: horizon_idx {h} out of range [0, {nT-1}]")
        fleet[vi, h:] += cnt

    return fleet  # shape (nV, nT)


def _make_variable_indexers(nR: int, nV: int, nT: int):
    """Return flat-index accessor closures and offsets for the four variable blocks.

    Variable layout in the flat vector of length ``n_tot``:
        x[r,v,t]  — dispatch counts (integer), shape (nR, nV, nT)
        y[r,t]    — shipped units (continuous), shape (nR, nT)
        s[r,t]    — leftover / carry-over stock (continuous), shape (nR, nT)
        u[r,v,t]  — empty capacity (continuous), shape (nR, nV, nT)

    Returns:
        Tuple of (n_tot, n_x, idx_y, idx_s, idx_u, ix, iy, is_, iu) where
        ix/iy/is_/iu are closures mapping (r[,v],t) → flat index.
    """
    n_x   = nR * nV * nT
    n_ys  = nR * nT
    n_u   = nR * nV * nT
    n_tot = n_x + 2 * n_ys + n_u

    idx_y = n_x
    idx_s = n_x + n_ys
    idx_u = n_x + 2 * n_ys

    def ix(r, v, t): return r * (nV * nT) + v * nT + t
    def iy(r, t):    return idx_y + r * nT + t
    def is_(r, t):   return idx_s + r * nT + t
    def iu(r, v, t): return idx_u + r * (nV * nT) + v * nT + t

    return n_tot, n_x, idx_y, idx_s, idx_u, ix, iy, is_, iu


def _build_milp_model(
    nR: int, nV: int, nT: int,
    D: "np.ndarray",
    caps: "np.ndarray",
    penalty_by_v: List[float],
    P_wait: float,
    economy_q: float,
    max_fleet: "np.ndarray",
    cost_vr: "np.ndarray",
    global_fleet: bool,
):
    """Build the MILP objective, bounds, and constraint matrix.

    Args:
        nR: Number of routes.
        nV: Number of vehicle types.
        nT: Number of planning horizons.
        D: Demand matrix of shape (nR, nT).
        caps: Vehicle capacities array of shape (nV,).
        penalty_by_v: Underload penalty per vehicle type, length nV.
        P_wait: Carry-over penalty per unit per horizon.
        economy_q: Minimum fill-rate ratio in [0, 1]; 0 disables the constraint.
        max_fleet: Available fleet matrix of shape (nV, nT).
        cost_vr: Trip cost matrix of shape (nV, nR).
        global_fleet: If True, use cumulative fleet limits (long trips);
            otherwise enforce per-horizon fleet limits (short trips).

    Returns:
        Tuple of (c, integrality, lb, ub, A_sp, lb_c_arr, ub_c_arr) ready
        for ``scipy.optimize.milp``.
    """
    n_tot, n_x, _, _, _, ix, iy, is_, iu = _make_variable_indexers(nR, nV, nT)

    # Objective vector
    c = np.zeros(n_tot)
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                c[ix(r, v, t)] = cost_vr[v, r]
                c[iu(r, v, t)] = penalty_by_v[v]
        for t in range(nT):
            c[is_(r, t)] = P_wait

    # Integrality (x is integer, rest continuous)
    integrality = np.zeros(n_tot)
    integrality[:n_x] = 1

    # Variable bounds
    lb = np.zeros(n_tot)
    ub = np.full(n_tot, np.inf)
    for vi in range(nV):
        for r in range(nR):
            for t in range(nT):
                ub[ix(r, vi, t)] = max_fleet[vi, t]
                ub[iu(r, vi, t)] = caps[vi]

    # Constraint rows
    A_rows: List[np.ndarray] = []
    lb_c:   List[float]      = []
    ub_c:   List[float]      = []

    # (1) Flow balance: y[r,t] + s[r,t] - s[r,t-1] = D[r,t]
    for r in range(nR):
        for t in range(nT):
            row = np.zeros(n_tot)
            row[iy(r, t)]  = 1.0
            row[is_(r, t)] = 1.0
            if t > 0:
                row[is_(r, t - 1)] = -1.0
            rhs = D[r, t]
            A_rows.append(row); lb_c.append(rhs); ub_c.append(rhs)

    # (2) Capacity link: Σ_v Cap_v·x[r,v,t] - y[r,t] - Σ_v u[r,v,t] = 0
    for r in range(nR):
        for t in range(nT):
            row = np.zeros(n_tot)
            for v in range(nV):
                row[ix(r, v, t)] =  caps[v]
                row[iu(r, v, t)] = -1.0
            row[iy(r, t)] = -1.0
            A_rows.append(row); lb_c.append(0.0); ub_c.append(0.0)

    # (2b) Underload only for dispatched vehicles: u[r,v,t] ≤ Cap_v * x[r,v,t]
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                row[iu(r, v, t)] =  1.0
                row[ix(r, v, t)] = -caps[v]
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(0.0)

    # (2c) Fill-rate floor: y[r,t] - economy_q * Σ_v cap_v*x[r,v,t] ≥ 0
    if economy_q > 0.0:
        for r in range(nR):
            for t in range(nT):
                row = np.zeros(n_tot)
                row[iy(r, t)] = 1.0
                for v in range(nV):
                    row[ix(r, v, t)] = -economy_q * caps[v]
                A_rows.append(row); lb_c.append(0.0); ub_c.append(np.inf)

    # (3) Fleet limits
    if global_fleet:
        # Cumulative dispatches up to horizon t cannot exceed max_fleet[v,t]
        for vi in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                for r in range(nR):
                    for tau in range(t + 1):
                        row[ix(r, vi, tau)] = 1.0
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(max_fleet[vi, t])
    else:
        # Per-horizon fleet: Σ_r x[r,v,t] ≤ max_fleet[v,t]
        for vi in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                for r in range(nR):
                    row[ix(r, vi, t)] = 1.0
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(max_fleet[vi, t])

    A_sp = csr_matrix(np.array(A_rows))
    return c, integrality, lb, ub, A_sp, np.array(lb_c), np.array(ub_c)


def _extract_milp_solution(
    x_vec: "np.ndarray",
    nR: int, nV: int, nT: int,
) -> tuple:
    """Unpack the flat MILP solution vector into (X, Y, S, U) matrices.

    Args:
        x_vec: Flat solution vector returned by ``scipy.optimize.milp``.
        nR: Number of routes.
        nV: Number of vehicle types.
        nT: Number of planning horizons.

    Returns:
        Tuple (X, Y, S, U) where:
            X: int dispatch counts, shape (nR, nV, nT)
            Y: shipped units, shape (nR, nT)
            S: leftover stock, shape (nR, nT)
            U: empty capacity, shape (nR, nV, nT)
    """
    _, _, _, _, _, ix, iy, is_, iu = _make_variable_indexers(nR, nV, nT)

    X = np.zeros((nR, nV, nT))
    Y = np.zeros((nR, nT))
    S = np.zeros((nR, nT))
    U = np.zeros((nR, nV, nT))
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                X[r, v, t] = round(x_vec[ix(r, v, t)])
        for t in range(nT):
            Y[r, t] = max(0.0, x_vec[iy(r, t)])
            S[r, t] = max(0.0, x_vec[is_(r, t)])
            for v in range(nV):
                U[r, v, t] = max(0.0, x_vec[iu(r, v, t)])
    return X, Y, S, U


def solve_irp_milp(
    demands: Dict[str, List[float]],
    vehicles_cfg: Dict,
    route_distances: Optional[Dict[str, float]] = None,
    global_fleet: bool = True,
    incoming_vehicles: Optional[List[Dict]] = None,
    granularity: float = 2.0,
) -> Dict:
    """Solve the IRP MILP jointly across all routes and planning horizons.

    Args:
        demands: Mapping ``{route_id: [D_t0, D_t1, D_t2, D_t3]}`` where D_t0 is
            current init stock and D_t1/t2/t3 are ML demand forecasts.
        vehicles_cfg: Contents of vehicles.json (vehicles list + penalties).
        route_distances: Optional ``{route_id: km}`` override.
        global_fleet: If True, cumulative fleet constraint (long trip — vehicle
            is occupied for the full planning window); if False, per-horizon
            constraint (short trip — vehicle can be reused each horizon).
        incoming_vehicles: List of ``{horizon_idx, vehicle_type, count}`` records
            describing vehicles arriving during the planning window.

    Returns:
        Dict with keys route_ids, X, Y, S, U, D, cost_vr, caps, max_fleet,
        penalty_by_v, P_wait for consumption by ``build_plan``.

    Raises:
        RuntimeError: If the MILP solver fails to find a feasible solution.
    """
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    nV = len(vehicles)
    caps = np.array([v["capacity_units"] for v in vehicles], dtype=float)
    horizons_list, period_minutes = make_horizons(granularity)
    nT = len(horizons_list)
    max_fleet = _build_fleet_limits(vehicles, incoming_vehicles, nT)

    penalty_by_v = [float(v["underload_penalty"]) for v in vehicles]
    P_wait = float(vehicles_cfg.get("wait_penalty_per_minute", 0)) * period_minutes
    economy_q = float(vehicles_cfg.get("economy_threshold", 0)) / 100.0
    economy_q = max(0.0, min(1.0, economy_q))

    route_ids = list(demands.keys())
    nR = len(route_ids)

    cost_vr = np.zeros((nV, nR))
    for vi, v in enumerate(vehicles):
        for ri, rid in enumerate(route_ids):
            dist = (route_distances or {}).get(rid, 15.0)
            fixed_dispatch = float(v.get("fixed_dispatch_cost") or 0)
            cost_vr[vi, ri] = v.get("cost_per_km", 0) * dist + fixed_dispatch

    D = np.array([[float(demands[rid][t]) for t in range(nT)] for rid in route_ids])

    c, integrality, lb, ub, A_sp, lb_c_arr, ub_c_arr = _build_milp_model(
        nR, nV, nT, D, caps, penalty_by_v, P_wait, economy_q, max_fleet, cost_vr, global_fleet
    )
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        constraints=LinearConstraint(A_sp, lb=lb_c_arr, ub=ub_c_arr),
        options={"time_limit": 30.0, "disp": False},
    )
    if not result.success and result.x is None:
        raise RuntimeError(f"MILP solver failed: {result.message}")

    X, Y, S, U = _extract_milp_solution(result.x, nR, nV, nT)
    return {
        "route_ids": route_ids,
        "X": X, "Y": Y, "S": S, "U": U,
        "D": D, "cost_vr": cost_vr, "caps": caps,
        "max_fleet": max_fleet,
        "penalty_by_v": np.array(penalty_by_v),
        "P_wait": P_wait,
    }


def _compute_effective_demands(
    demands: Dict[str, List[float]],
    conformal_margins: Optional[Dict[str, List[float]]],
) -> Dict[str, List[int]]:
    """Apply upper-CI conformal margins to point-forecast demands for the MILP input.

    Horizon 0 (init stock) is never inflated. Horizons 1–3 are rounded to integer
    because physical goods are discrete units and the MILP flow-balance gives
    integer-valued y/s only when demand D is integer.

    Args:
        demands: Point-forecast demands ``{route_id: [D0, D1, D2, D3]}``.
        conformal_margins: Per-route, per-horizon margin offsets, or None for no
            inflation.

    Returns:
        Effective integer demands suitable for ``solve_irp_milp``.
    """
    if conformal_margins:
        return {
            rid: [
                round(d + (conformal_margins.get(rid, [0.0] * len(dlist))[i] if i > 0 else 0.0))
                for i, d in enumerate(dlist)
            ]
            for rid, dlist in demands.items()
        }
    return {rid: [round(d) for d in dlist] for rid, dlist in demands.items()}


def _build_horizon_rows(
    r_idx: int,
    rid: str,
    res: Dict,
    demands: Dict[str, List[float]],
    conformal_margins: Optional[Dict[str, List[float]]],
    v_names: List[str],
    timestamp: str,
    horizons_list: Optional[List[tuple]] = None,
) -> List[Dict]:
    """Build plan row dicts for all horizons of one route.

    Produces one row per dispatched vehicle type per horizon, plus one
    zero-vehicle row for horizons with no dispatch.

    Args:
        r_idx: Route index in the MILP result arrays.
        rid: Route identifier string.
        res: Dict returned by ``solve_irp_milp`` (contains X, Y, S, U, D, ...).
        demands: Original point-forecast demands (without margin inflation).
        conformal_margins: Per-route conformal margin lists, or None.
        v_names: Vehicle type name list (same order as in vehicles_cfg).
        timestamp: Timestamp string to embed in each row.

    Returns:
        List of row dicts, one per (horizon, vehicle_type) combination.
    """
    X, Y, S, U = res["X"], res["Y"], res["S"], res["U"]
    D, cost_vr = res["D"], res["cost_vr"]
    penalty_by_v = res["penalty_by_v"]
    P_wait = res["P_wait"]
    nV = len(v_names)

    rows = []
    _horizons = horizons_list if horizons_list is not None else HORIZONS
    for t_idx, (label, _) in enumerate(_horizons):
        y_rt  = float(Y[r_idx, t_idx])
        s_rt  = float(S[r_idx, t_idx])
        u_vec = [float(U[r_idx, vi, t_idx]) for vi in range(nV)]
        u_sum = sum(u_vec)
        d_rt  = float(D[r_idx, t_idx])
        d_point = round(float(demands[rid][t_idx])) if rid in demands else round(d_rt)
        margins_list = conformal_margins.get(rid, [0.0] * (t_idx + 1)) if conformal_margins else None
        m = (margins_list[t_idx] if margins_list and t_idx < len(margins_list) else 0.0)
        d_lower = round(max(0.0, d_point - m))
        d_upper = round(d_point + m)
        s_prev = float(S[r_idx, t_idx - 1]) if t_idx > 0 else 0.0

        dispatched = [vi for vi in range(nV) if X[r_idx, vi, t_idx] > 0]
        cost_wait_total = s_rt * P_wait
        cost_wait_share = (cost_wait_total / len(dispatched)) if dispatched else cost_wait_total

        if dispatched:
            for vi in dispatched:
                cost_fixed_v = float(X[r_idx, vi, t_idx]) * cost_vr[vi, r_idx]
                cost_under_v = u_vec[vi] * penalty_by_v[vi]
                cost_total_v = cost_fixed_v + cost_under_v + cost_wait_share
                rows.append({
                    "route_id":              rid,
                    "timestamp":             timestamp,
                    "horizon":               label,
                    "vehicle_type":          v_names[vi],
                    "vehicles_count":        int(X[r_idx, vi, t_idx]),
                    "demand_new":            round(d_point),
                    "demand_lower":          d_lower,
                    "demand_upper":          d_upper,
                    "demand_carried_over":   round(s_prev),
                    "total_available":       round(d_rt + s_prev),
                    "actually_shipped":      round(y_rt),
                    "leftover_stock":        round(s_rt),
                    "empty_capacity_units":  round(u_vec[vi], 2),
                    "cost_fixed":            round(cost_fixed_v, 2),
                    "cost_underload":        round(cost_under_v, 2),
                    "cost_wait":             round(cost_wait_share, 2),
                    "cost_total":            round(cost_total_v, 2),
                })
        else:
            cost_under = u_sum * penalty_by_v.mean()
            rows.append({
                "route_id":              rid,
                "timestamp":             timestamp,
                "horizon":               label,
                "vehicle_type":          "none",
                "vehicles_count":        0,
                "demand_new":            round(d_point),
                "demand_lower":          d_lower,
                "demand_upper":          d_upper,
                "demand_carried_over":   round(s_prev),
                "total_available":       round(d_rt + s_prev),
                "actually_shipped":      0,
                "leftover_stock":        round(s_rt),
                "empty_capacity_units":  round(u_sum, 2),
                "cost_fixed":            0.0,
                "cost_underload":        round(cost_under, 2),
                "cost_wait":             round(cost_wait_total, 2),
                "cost_total":            round(cost_under + cost_wait_total, 2),
            })
    return rows


def build_plan(
    timestamp: str,
    demands: Dict[str, List[float]],
    vehicles_cfg: Dict,
    office_id: str = "",
    route_distances: Optional[Dict[str, float]] = None,
    global_fleet: bool = False,
    incoming_vehicles: Optional[List[Dict]] = None,
    conformal_margins: Optional[Dict[str, List[float]]] = None,
    granularity: float = 2.0,
) -> pd.DataFrame:
    """Assemble a dispatch plan DataFrame from the MILP solution.

    Args:
        timestamp: ISO timestamp string embedded in every output row.
        demands: Point-forecast demands ``{route_id: [D0, D1, D2, D3]}``.
        vehicles_cfg: Contents of vehicles.json.
        office_id: Office identifier prepended as a column when non-empty.
        route_distances: Optional ``{route_id: km}`` override for distance-based costs.
        global_fleet: Passed through to ``solve_irp_milp``; see that function.
        incoming_vehicles: Optional arriving-vehicle records; see ``_build_fleet_limits``.
        conformal_margins: ``{route_id: [m0, m1, m2, m3]}``; m0 must be 0.0 (horizon A is
            deterministic). Margins inflate effective demand for the MILP and are
            reflected in demand_lower/demand_upper columns.

    Returns:
        DataFrame with one row per (route_id, horizon, vehicle_type) combination.
        An ``office_from_id`` column is prepended when ``office_id`` is non-empty.
    """
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    v_names = [v["vehicle_type"] for v in vehicles]

    demands_effective = _compute_effective_demands(demands, conformal_margins)
    res = solve_irp_milp(demands_effective, vehicles_cfg, route_distances, global_fleet, incoming_vehicles, granularity)

    horizons_list, _ = make_horizons(granularity)

    rows = []
    for r_idx, rid in enumerate(res["route_ids"]):
        rows.extend(_build_horizon_rows(r_idx, rid, res, demands, conformal_margins, v_names, timestamp, horizons_list))

    df = pd.DataFrame(rows)
    if office_id and not df.empty:
        df.insert(0, "office_from_id", office_id)
    return df


def main():
    parser = argparse.ArgumentParser(description="IRP MILP dispatch optimizer (0-2/2-4/4-6h horizons)")
    route_grp = parser.add_mutually_exclusive_group(required=True)
    route_grp.add_argument("--route_id",  help="Один route_id")
    route_grp.add_argument("--route_ids", help="Несколько route_id через запятую")
    parser.add_argument("--timestamp",   required=True, help="Например: 2025-05-30 10:30:00")
    parser.add_argument("--train_path",  default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--models_dir",  default=DEFAULT_MODELS_DIR)
    parser.add_argument("--vehicles",    default="vehicles.json")
    parser.add_argument("--out",         default="", help="Путь для сохранения CSV")
    parser.add_argument("--global_fleet", action="store_true",
                        help="Глобальное ограничение парка (долгий рейс)")
    parser.add_argument("--incoming", default="",
                        help="Путь к incoming_vehicles.json (опционально)")
    args = parser.parse_args()

    route_ids = (
        [r.strip() for r in args.route_ids.split(",") if r.strip()]
        if args.route_ids
        else [args.route_id.strip()]
    )
    ts = args.timestamp

    vehicles_cfg = json.loads(Path(args.vehicles).read_text())

    incoming_vehicles: Optional[List[Dict]] = None
    if args.incoming:
        incoming_data = json.loads(Path(args.incoming).read_text())
        incoming_vehicles = incoming_data.get("incoming", [])

    office_map = load_route_office_map(Path(args.train_path))
    office_id  = office_map.get(route_ids[0], "")

    models             = load_models(Path(args.models_dir))
    X_all, feature_cols = prepare_feature_matrix(Path(args.train_path))

    demands: Dict[str, List[float]] = {}
    for rid in route_ids:
        preds = predict_for_route_timestamp(X_all, feature_cols, models,
                                            route_id=rid, timestamp=ts)
        demands[rid] = [
            0,                    # D_t0: текущий сток склада
            preds["pred_0_2h"],   # D_t1: ML-прогноз 0-2h
            preds["pred_2_4h"],   # D_t2: ML-прогноз 2-4h
            preds["pred_4_6h"],   # D_t3: ML-прогноз 4-6h
        ]

    plan = build_plan(
        timestamp=ts,
        demands=demands,
        vehicles_cfg=vehicles_cfg,
        office_id=office_id,
        global_fleet=args.global_fleet,
        incoming_vehicles=incoming_vehicles,
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(plan.to_string(index=False))

    total_leftover = plan.groupby(["route_id", "horizon"])["leftover_stock"].first().sum()
    print(f"\nTotal leftover stock across all routes/horizons: {total_leftover:.2f}")

    if args.out:
        plan.to_csv(args.out, index=False, float_format="%.2f")
        print(f"Saved → {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
