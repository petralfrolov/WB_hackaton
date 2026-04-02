"""Generate mock non-conformity calibration scores.

Produces data/non_conformity_scores.csv with columns:
  route_id, horizon, score

Scores simulate absolute residuals |y_actual - y_hat| on a held-out
calibration set.  Horizon-specific noise scales grow with forecast
distance (0-2h < 2-4h < 4-6h), matching typical ML behaviour.

Run from the backend/ directory:
    python scripts/generate_mock_ncs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

WAREHOUSE_META = Path(__file__).resolve().parent.parent / "warehouse_metadata.json"
NCS_PATH = Path(__file__).resolve().parent.parent / "data" / "non_conformity_scores.csv"

# (loc, scale) for np.random.default_rng.lognormal — gives right-skewed
# positive residuals that grow with forecast horizon.
# lognormal(mu, sigma) has median = exp(mu), mean = exp(mu + sigma^2/2)
HORIZON_PARAMS: dict[str, tuple[float, float]] = {
    "0-2h": (0.55, 0.40),   # median ≈ 1.7 units,  90th-pct ≈ 3.5
    "2-4h": (1.10, 0.45),   # median ≈ 3.0 units,  90th-pct ≈ 6.5
    "4-6h": (1.55, 0.50),   # median ≈ 4.7 units,  90th-pct ≈ 10.5
}

N_CAL = 25   # calibration observations per (route, horizon)


def main() -> None:
    meta = json.loads(WAREHOUSE_META.read_text(encoding="utf-8"))
    route_ids: list[str] = []
    for w in meta["warehouses"]:
        route_ids.extend(str(r) for r in w["route_ids"])

    NCS_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for route_id in route_ids:
        for horizon, (mu, sigma) in HORIZON_PARAMS.items():
            # Deterministic seed derived from route_id and horizon string so
            # the file is reproducible yet distinct per (route, horizon).
            seed = (int(route_id) * 7919 + sum(ord(c) for c in horizon)) % (2 ** 32)
            rng = np.random.default_rng(seed)
            scores = rng.lognormal(mean=mu, sigma=sigma, size=N_CAL)
            for s in scores:
                rows.append({"route_id": route_id, "horizon": horizon, "score": round(float(s), 4)})

    with open(NCS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "horizon", "score"])
        writer.writeheader()
        writer.writerows(rows)

    n_routes = len(route_ids)
    n_horizons = len(HORIZON_PARAMS)
    print(f"Generated {len(rows)} rows  "
          f"({n_routes} routes × {n_horizons} horizons × {N_CAL} cal points)")
    print(f"Saved → {NCS_PATH}")

    # Quick sanity print
    import pandas as pd
    df = pd.read_csv(NCS_PATH)
    print("\nMedian score by horizon:")
    print(df.groupby("horizon")["score"].median().round(2).to_string())


if __name__ == "__main__":
    main()
