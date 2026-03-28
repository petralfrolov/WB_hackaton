"""Main pipeline: load - features - split - train - evaluate - submit."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    TRACK, CONFIG, TARGET_COL, FUTURE_TARGET_COLS,
    LGB_PARAMS, RANDOM_STATE,
)
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import (train_lgb_models, train_catboost_models, train_dirrec_models,
                   predict_steps, predict_steps_catboost, predict_steps_dirrec,
                   compute_bias_factor, build_submission)

EXPERIMENTS_LOG = "experiments.json"


def log_experiment(name: str, params: dict, wape: float, rbias: float, total: float):
    """Append experiment result to JSON log."""
    log_path = Path(EXPERIMENTS_LOG)
    if log_path.exists():
        history = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        history = []
    history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "params": params,
        "wape": round(wape, 6),
        "rbias": round(rbias, 6),
        "total": round(total, 6),
    })
    log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Logged to {EXPERIMENTS_LOG}")


def run(experiment_name="baseline_lgb", lgb_params=None, use_bias_correction=True,
        log_target=False, extended_features=False, use_catboost=False, cb_params=None,
        route_normalize=False, use_dirrec=False):
    t0 = time.time()
    metric = WapePlusRbias()

    # 1. Load
    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)
    train_df, test_df = load_data()

    # 2. Feature engineering
    print("\n[Features]")
    train_df = make_features(train_df, extended=extended_features)
    print(f"  Columns after FE: {train_df.shape[1]}")

    # 3. Future targets
    train_df = create_future_targets(train_df)

    # 4. Build feature list & split
    feature_cols = build_feature_cols(train_df)
    print(f"  Feature count: {len(feature_cols)}")

    X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)

    # 5. Encode categoricals
    X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
    print(f"  Categorical: {cat_features}")

    # 5b. Log-transform target if requested
    if log_target:
        print("  [log1p target transform enabled]")
        y_fit_train = np.log1p(y_fit.clip(lower=0))
        y_valid_train = np.log1p(y_valid.clip(lower=0))
    else:
        y_fit_train = y_fit
        y_valid_train = y_valid

    # 5c. Route-level normalization
    # Compute per-route scale factors from fit set, apply to all splits
    route_scale = None
    if route_normalize:
        print("  [route-level target normalization enabled]")
        # scale = mean target per route (on fit set), indexed by route_id
        fit_route_ids = X_fit["route_id"].astype(str)
        route_scale = (
            pd.Series(y_fit_train.values.mean(axis=1), index=fit_route_ids)
            .groupby(level=0).mean()
            + 1e-6
        )
        def _scale_y(y_df, X_df):
            rids = X_df["route_id"].astype(str)
            scale = rids.map(route_scale).fillna(route_scale.mean()).values.reshape(-1, 1)
            return pd.DataFrame(y_df.values / scale, columns=y_df.columns, index=y_df.index), scale

        y_fit_train, fit_scale  = _scale_y(y_fit_train, X_fit)
        y_valid_train, val_scale = _scale_y(y_valid_train, X_valid)

    # 6. Train
    print("\n[Training]")
    params = lgb_params or LGB_PARAMS
    if use_catboost:
        models = train_catboost_models(X_fit, y_fit_train, X_valid, y_valid_train,
                                       cat_features=cat_features, cb_params=cb_params)
    elif use_dirrec:
        models = train_dirrec_models(X_fit, y_fit_train, X_valid, y_valid_train, lgb_params=params)
        # Discard X_valid_aug — predict_steps_dirrec builds features sequentially from scratch
        models.pop("_X_valid_aug", None)
    else:
        models = train_lgb_models(X_fit, y_fit_train, X_valid, y_valid_train, lgb_params=params)

    # 7. Predict & evaluate
    print("\n[Evaluation]")
    if use_catboost:
        fit_pred = predict_steps_catboost(models, X_fit, cat_features)
        valid_pred = predict_steps_catboost(models, X_valid, cat_features)
    elif use_dirrec:
        fit_pred = predict_steps_dirrec(models, X_fit)
        valid_pred = predict_steps_dirrec(models, X_valid)
    else:
        fit_pred = predict_steps(models, X_fit)
        valid_pred = predict_steps(models, X_valid)

    # Inverse log transform if needed
    if log_target:
        fit_pred = pd.DataFrame(np.expm1(fit_pred.clip(lower=0)), columns=fit_pred.columns, index=fit_pred.index)
        valid_pred = pd.DataFrame(np.expm1(valid_pred.clip(lower=0)), columns=valid_pred.columns, index=valid_pred.index)
        fit_pred = fit_pred.clip(lower=0)
        valid_pred = valid_pred.clip(lower=0)

    # Inverse route normalization
    if route_normalize and route_scale is not None:
        def _unscale(pred_df, X_df):
            rids = X_df["route_id"].astype(str)
            scale = rids.map(route_scale).fillna(route_scale.mean()).values.reshape(-1, 1)
            return pd.DataFrame(pred_df.values * scale, columns=pred_df.columns, index=pred_df.index).clip(lower=0)
        fit_pred = _unscale(fit_pred, X_fit)
        valid_pred = _unscale(valid_pred, X_valid)

    w_fit, r_fit, t_fit = metric.calculate_components(y_fit, fit_pred)
    w_val, r_val, t_val = metric.calculate_components(y_valid, valid_pred)
    print(f"  FIT   — WAPE: {w_fit:.4f}  RBias: {r_fit:.4f}  Total: {t_fit:.4f}")
    print(f"  VALID — WAPE: {w_val:.4f}  RBias: {r_val:.4f}  Total: {t_val:.4f}")

    # 8. Bias correction
    bias_factor = 1.0
    if use_bias_correction:
        bias_factor = compute_bias_factor(y_valid, valid_pred)
        corrected_valid = (valid_pred * bias_factor).clip(lower=0)
        w_vc, r_vc, t_vc = metric.calculate_components(y_valid, corrected_valid)
        print(f"  VALID (bias-corrected, factor={bias_factor:.4f}) — "
              f"WAPE: {w_vc:.4f}  RBias: {r_vc:.4f}  Total: {t_vc:.4f}")
        final_wape, final_rbias, final_total = w_vc, r_vc, t_vc
    else:
        final_wape, final_rbias, final_total = w_val, r_val, t_val

    # 9. Submission
    print("\n[Submission]")
    if use_catboost:
        test_pred = predict_steps_catboost(models, X_test, cat_features)
    elif use_dirrec:
        test_pred = predict_steps_dirrec(models, X_test)
    else:
        test_pred = predict_steps(models, X_test)
    if log_target:
        test_pred = pd.DataFrame(np.expm1(test_pred.clip(lower=0)), columns=test_pred.columns, index=test_pred.index)
        test_pred = test_pred.clip(lower=0)
    if route_normalize and route_scale is not None:
        test_pred = _unscale(test_pred, X_test)
    submission = build_submission(test_pred, X_test, inference_ts, test_df, bias_factor=bias_factor)
    sub_path = f"submission_{TRACK}_{experiment_name}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}  ({len(submission)} rows)")

    # 10. Log
    log_experiment(
        experiment_name,
        {k: v for k, v in params.items() if k not in ("n_jobs", "verbose", "random_state")},
        final_wape, final_rbias, final_total,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.  VALID metric = {final_total:.4f}")
    return final_total


if __name__ == "__main__":
    run()
