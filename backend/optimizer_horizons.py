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

from ml_prediction import (
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


def load_route_office_map(train_path: Path) -> Dict[str, str]:
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
        if not (0 <= h < nT):
            raise ValueError(f"incoming_vehicles: horizon_idx {h} out of range [0, {nT-1}]")
        fleet[vi, h:] += cnt

    return fleet  # shape (nV, nT)


def solve_irp_milp(
    demands: Dict[str, List[float]],
    vehicles_cfg: Dict,
    route_distances: Optional[Dict[str, float]] = None,
    global_fleet: bool = True,
    incoming_vehicles: Optional[List[Dict]] = None,
) -> Dict:
    """Решить IRP MILP глобально по всем маршрутам и горизонтам.

    Parameters
    ----------
    demands : {route_id: [D_t0, D_t1, D_t2, D_t3]}
        D_t0 — текущий сток на складе; D_t1/2/3 — ML-прогноз спроса по горизонтам.
    vehicles_cfg : содержимое vehicles.json.
    route_distances : опциональный словарь {route_id: km} для переопределения дистанции.
    global_fleet : True → Σ_{r,t} x ≤ MaxV_v[t] (долгий рейс, ТС занята весь горизонт);
                   False → Σ_r x[r,v,t] ≤ MaxV_v[t] отдельно для каждого t (короткий рейс).
    incoming_vehicles : список {horizon_idx, vehicle_type, count} из incoming_vehicles.json.
        ТС, прибывающие к складу в ходе планового окна; увеличивают лимит парка
        начиная с указанного горизонта.
    """
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    nV = len(vehicles)
    caps = np.array([v["capacity_units"] for v in vehicles], dtype=float)
    nT = len(HORIZONS)
    # max_fleet[v, t] — кол-во ТС типа v, доступных в горизонте t (с учётом прибытий)
    max_fleet = _build_fleet_limits(vehicles, incoming_vehicles, nT)  # (nV, nT)

    penalty_by_v = [float(v["underload_penalty"]) for v in vehicles]
    P_wait = float(vehicles_cfg.get("wait_penalty_per_minute", 0)) * PERIOD_MINUTES
    # economy_threshold: minimum fill-rate ratio in (0, 1]; 0 = no constraint
    economy_q = float(vehicles_cfg.get("economy_threshold", 0)) / 100.0
    economy_q = max(0.0, min(1.0, economy_q))

    default_dist = 15.0
    route_ids = list(demands.keys())
    nR = len(route_ids)

    # cost_vr[v, r] = стоимость одного рейса ТС типа v по маршруту r
    cost_vr = np.zeros((nV, nR))
    for vi, v in enumerate(vehicles):
        for ri, rid in enumerate(route_ids):
            dist = (route_distances or {}).get(rid, default_dist)
            fixed_dispatch = float(v.get("fixed_dispatch_cost") or 0)
            cost_vr[vi, ri] = v.get("cost_per_km", 0) * dist + fixed_dispatch

    # D[r, t] — матрица спроса
    D = np.array([[float(demands[rid][t]) for t in range(nT)] for rid in route_ids])

    # ---- Индексирование переменных ----
    # x[r,v,t]  → [0 .. n_x)
    # y[r,t]    → [n_x .. n_x + n_ys)
    # s[r,t]    → [n_x + n_ys .. n_x + 2*n_ys)
    # u[r,v,t]  → [n_x + 2*n_ys .. n_x + 2*n_ys + n_u)
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

    # ---- Целевая функция ----
    c = np.zeros(n_tot)
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                c[ix(r, v, t)] = cost_vr[v, r]
                c[iu(r, v, t)] = penalty_by_v[v]
        for t in range(nT):
            c[is_(r, t)] = P_wait

    # ---- Целочисленность: x — целые, y/s/u — непрерывные ----
    integrality = np.zeros(n_tot)
    integrality[:n_x] = 1

    # ---- Границы переменных ----
    lb = np.zeros(n_tot)
    ub = np.full(n_tot, np.inf)
    for vi in range(nV):
        for r in range(nR):
            for t in range(nT):
                ub[ix(r, vi, t)] = max_fleet[vi, t]
                ub[iu(r, vi, t)] = caps[vi]

    # ---- Ограничения ----
    A_rows: List[np.ndarray] = []
    lb_c:   List[float]      = []
    ub_c:   List[float]      = []

    # (1) Баланс потока: y[r,t] + s[r,t] − s[r,t−1] = D[r,t]
    for r in range(nR):
        for t in range(nT):
            row = np.zeros(n_tot)
            row[iy(r, t)]  = 1.0
            row[is_(r, t)] = 1.0
            if t > 0:
                row[is_(r, t - 1)] = -1.0
            rhs = D[r, t]
            A_rows.append(row); lb_c.append(rhs); ub_c.append(rhs)

    # (2) Связь вместимости: Σ_v Cap_v·x[r,v,t] − y[r,t] − Σ_v u[r,v,t] = 0
    for r in range(nR):
        for t in range(nT):
            row = np.zeros(n_tot)
            for v in range(nV):
                row[ix(r, v, t)] = caps[v]
                row[iu(r, v, t)] = -1.0
            row[iy(r, t)] = -1.0
            A_rows.append(row); lb_c.append(0.0); ub_c.append(0.0)

    # (2b) u[r,v,t] ≤ Cap_v * x[r,v,t]  (недогруз только у отправленных ТС)
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                row[iu(r, v, t)] = 1.0
                row[ix(r, v, t)] = -caps[v]
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(0.0)

    # (2c) Fill-rate: y[r,t] ≥ economy_q * Σ_v cap_v*x[r,v,t]  for all t
    #      Rearranged: y[r,t] - economy_q * Σ_v cap_v*x[r,v,t] ≥ 0
    if economy_q > 0.0:
        for r in range(nR):
            for t in range(nT):
                row = np.zeros(n_tot)
                row[iy(r, t)] = 1.0
                for v in range(nV):
                    row[ix(r, v, t)] = -economy_q * caps[v]
                A_rows.append(row); lb_c.append(0.0); ub_c.append(np.inf)

    # (3) Общий автопарк с учётом прибывающих ТС
    if global_fleet:
        # Для каждого горизонта t накопленная сумма отправленных ТС (от 0 до t включительно)
        # не должна превышать флот, доступный именно в момент t.
        # Это исключает отправку машин, которые физически ещё не прибыли на склад.
        for vi in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                for r in range(nR):
                    for tau in range(t + 1):
                        row[ix(r, vi, tau)] = 1.0
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(max_fleet[vi, t])
    else:
        # Σ_r x[r,v,t] ≤ MaxV_v[v,t]  для каждого (v, t)  (короткий рейс)
        for vi in range(nV):
            for t in range(nT):
                row = np.zeros(n_tot)
                for r in range(nR):
                    row[ix(r, vi, t)] = 1.0
                A_rows.append(row); lb_c.append(-np.inf); ub_c.append(max_fleet[vi, t])

    A_sp = csr_matrix(np.array(A_rows))
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        constraints=LinearConstraint(A_sp, lb=np.array(lb_c), ub=np.array(ub_c)),
    )

    if not result.success:
        raise RuntimeError(f"MILP solver failed: {result.message}")

    xv = result.x
    X = np.zeros((nR, nV, nT))
    Y = np.zeros((nR, nT))
    S = np.zeros((nR, nT))
    U = np.zeros((nR, nV, nT))
    for r in range(nR):
        for v in range(nV):
            for t in range(nT):
                X[r, v, t] = round(xv[ix(r, v, t)])
        for t in range(nT):
            Y[r, t] = max(0.0, xv[iy(r, t)])
            S[r, t] = max(0.0, xv[is_(r, t)])
            for v in range(nV):
                U[r, v, t] = max(0.0, xv[iu(r, v, t)])

    return {
        "route_ids": route_ids,
        "X": X, "Y": Y, "S": S, "U": U,
        "D": D, "cost_vr": cost_vr, "caps": caps,
        "max_fleet": max_fleet,
        "penalty_by_v": np.array(penalty_by_v),
        "P_wait": P_wait,
    }


def build_plan(
    timestamp: str,
    demands: Dict[str, List[float]],
    vehicles_cfg: Dict,
    office_id: str = "",
    route_distances: Optional[Dict[str, float]] = None,
    global_fleet: bool = False,
    incoming_vehicles: Optional[List[Dict]] = None,
    conformal_margins: Optional[Dict[str, List[float]]] = None,
) -> pd.DataFrame:
    """Собрать план отправок в DataFrame из решения MILP.

    conformal_margins: optional dict {route_id: [m_t0, m_t1, m_t2, m_t3]}.
    m_t0 should be 0 (init_stock is deterministic); m_t1/t2/t3 are the
    per-horizon conformal margins.  When provided, effective demand used in
    the MILP upper-bounds each horizon by pred + margin (conservative).
    demand_lower/demand_upper in the output reflect the full CI per row.
    """
    vehicles = vehicles_cfg.get("vehicles", vehicles_cfg)
    v_names = [v["vehicle_type"] for v in vehicles]
    nV = len(vehicles)

    # Build adjusted demands for MILP (use upper CI bound as effective demand)
    # Demands are rounded to integers: physical goods are discrete units, and the
    # MILP balance y+s = D gives integer y/s only when D is integer.
    if conformal_margins:
        demands_effective = {
            rid: [
                round(d + (conformal_margins.get(rid, [0.0, 0.0, 0.0, 0.0])[i] if i > 0 else 0.0))
                for i, d in enumerate(dlist)
            ]
            for rid, dlist in demands.items()
        }
    else:
        demands_effective = {
            rid: [round(d) for d in dlist]
            for rid, dlist in demands.items()
        }

    res = solve_irp_milp(demands_effective, vehicles_cfg, route_distances, global_fleet, incoming_vehicles)
    route_ids = res["route_ids"]
    X, Y, S, U   = res["X"], res["Y"], res["S"], res["U"]
    D, cost_vr   = res["D"], res["cost_vr"]
    penalty_by_v  = res["penalty_by_v"]
    P_wait       = res["P_wait"]

    rows = []
    for r_idx, rid in enumerate(route_ids):
        for t_idx, (label, _) in enumerate(HORIZONS):
            y_rt  = float(Y[r_idx, t_idx])
            s_rt  = float(S[r_idx, t_idx])
            u_vec = [float(U[r_idx, vi, t_idx]) for vi in range(nV)]
            u_sum = sum(u_vec)
            d_rt  = float(D[r_idx, t_idx])  # effective demand (may include margin)
            # point forecast demand (original, without margin) — rounded to int (goods are discrete)
            d_point = round(float(demands[rid][t_idx])) if rid in demands else round(d_rt)
            m = (
                conformal_margins.get(rid, [0.0, 0.0, 0.0, 0.0])[t_idx]
                if conformal_margins else 0.0
            )
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
                # горизонт без отправки — сохраняем строку с нулевым флотом
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
