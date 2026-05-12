"""
LightGBM-based per-core thermal delta predictor.

Replaces the Ridge model in the original script while keeping the same CSV
format, same data cleaning, same phase logic, and the same evaluation metrics.

Key differences from the Ridge version:
- Tree models do not extrapolate insanely outside the training range,
  which directly addresses the catastrophic over/underestimation we saw
  in the 100+ C region.
- No hand-crafted interaction features. The tree learns interactions itself.
- Phase is a categorical feature, not a one-hot with explicit interactions.
- No StandardScaler on X or Y. Trees are scale invariant.
- One LightGBM regressor per core (4 total). Each core has its own thermal
  behavior, and independent models tend to be more accurate per target.
- Sample weighting boosts high-temperature samples so the model stops
  systematically underestimating peaks.
- C++ export writes the raw tree structure (split feature, threshold,
  children, leaf values) so the C++ scheduler only needs a ~50 line
  tree traversal function. No LightGBM runtime dependency.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

NUM_CORES = 4

PHASE_NAMES = [
    "master_only",
    "workers_only",
]

PHASE_TO_INT = {name: i for i, name in enumerate(PHASE_NAMES)}


def parse_vec(value, expected_len=NUM_CORES, colname=""):
    if pd.isna(value):
        raise ValueError(f"Missing value in column {colname}")

    parts = str(value).strip().strip('"').split(";")
    vals = np.array([float(x) for x in parts], dtype=np.float32)

    if len(vals) != expected_len:
        raise ValueError(
            f"Column {colname} expected {expected_len} values, got {len(vals)}: {value}"
        )

    return vals


def safe_log_ips(ips):
    ips = np.asarray(ips, dtype=np.float32)
    ips = np.maximum(ips, 0.0)
    return np.log10(ips + 1.0)


def cpi_to_ipc(cpi):
    cpi = np.asarray(cpi, dtype=np.float32)
    ipc = np.zeros_like(cpi)

    valid = cpi < 100.0
    ipc[valid] = 1.0 / np.maximum(cpi[valid], 1e-6)

    return ipc


def neighbor_mean(x):
    x = np.asarray(x, dtype=np.float32)
    return (np.sum(x) - x) / 3.0


def classify_phase(start_util):
    """
    Same logic as the Ridge version. Returns None for transition intervals
    so they are dropped from training.
    """
    u = np.asarray(start_util, dtype=np.float32)

    transition_mask = (u >= 0.01) & (u <= 0.30)
    if np.any(transition_mask):
        return None

    active_mask = u > 0.30
    worker_count = int(np.sum(active_mask[1:]))

    if worker_count > 0:
        return "workers_only"

    return "master_only"


def row_has_invalid_temperature(row):
    if float(row["start_peak_temp_c"]) < 0:
        return True

    start_temp = parse_vec(row["start_core_temps_c"], colname="start_core_temps_c")
    end_temp = parse_vec(row["end_core_temps_c"], colname="end_core_temps_c")

    if np.any(start_temp < 0):
        return True

    if np.any(end_temp < 0):
        return True

    return False


# ------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------

@dataclass
class DatasetBundle:
    X: np.ndarray
    Y: np.ndarray
    start_temps: np.ndarray
    phases: np.ndarray
    feature_names: list
    target_names: list
    categorical_feature_indices: list
    df: pd.DataFrame


def build_dataset(csv_path):
    """
    Build a *much smaller* feature set than the Ridge version.

    Tree models learn interactions themselves, so we drop:
    - All phase x feature interaction terms
    - All temp_power / temp_freq / temp_workload products
    - All polynomial frequency terms (f^2, f^3) -- the tree can pick splits
      that approximate any monotonic transformation

    We keep:
    - Per-core scalars (one row per sample, 4 cores collapsed into per-core
      columns c0..c3)
    - A few cheap aggregates (sum/max) because they help splits in shallow trees
    - workload_index because it is a meaningful composite physical quantity
    - phase as a single integer categorical feature
    """
    raw_df = pd.read_csv(csv_path)

    required_cols = [
        "start_peak_temp_c",
        "old_frequencies_mhz",
        "new_frequencies_mhz",
        "active_cores",
        "start_core_temps_c",
        "end_core_temps_c",
        "start_core_powers_w",
        "start_core_utilizations",
        "start_core_cpis",
        "start_core_rel_nuca_cpis",
        "start_core_ips",
    ]

    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    X_rows = []
    Y_rows = []
    start_temp_rows = []
    phase_rows = []
    kept_indices = []

    feature_names = None

    skipped_invalid_temp = 0
    skipped_invalid_ips = 0
    skipped_parse_error = 0
    skipped_transition_phase = 0

    for idx, row in raw_df.iterrows():
        try:
            if row_has_invalid_temperature(row):
                skipped_invalid_temp += 1
                continue

            old_f = parse_vec(row["old_frequencies_mhz"], colname="old_frequencies_mhz") / 1000.0
            new_f = parse_vec(row["new_frequencies_mhz"], colname="new_frequencies_mhz") / 1000.0
            delta_f = new_f - old_f

            active = parse_vec(row["active_cores"], colname="active_cores")

            start_temp = parse_vec(row["start_core_temps_c"], colname="start_core_temps_c")
            end_temp = parse_vec(row["end_core_temps_c"], colname="end_core_temps_c")
            delta_temp = end_temp - start_temp

            start_power = parse_vec(row["start_core_powers_w"], colname="start_core_powers_w")
            start_util = parse_vec(row["start_core_utilizations"], colname="start_core_utilizations")

            start_cpi = parse_vec(row["start_core_cpis"], colname="start_core_cpis")
            start_ipc = cpi_to_ipc(start_cpi)

            start_rel_nuca_cpi = parse_vec(
                row["start_core_rel_nuca_cpis"],
                colname="start_core_rel_nuca_cpis",
            )

            start_ips = parse_vec(row["start_core_ips"], colname="start_core_ips")
            if np.any(start_ips < 0):
                skipped_invalid_ips += 1
                continue
            log_start_ips = safe_log_ips(start_ips)

            start_peak = float(row["start_peak_temp_c"])

            phase = classify_phase(start_util)
            if phase is None:
                skipped_transition_phase += 1
                continue

            temp_neighbor = neighbor_mean(start_temp)
            power_neighbor = neighbor_mean(start_power)

            workload_index = active * start_util * (new_f ** 3) * start_ipc

            features = []
            names = []

            # Global / aggregate features. Trees benefit from these because
            # a single split on e.g. max_start_temp can replace many splits
            # on individual core temps.
            globals_dict = {
                "start_peak_temp": start_peak,
                "max_start_temp": np.max(start_temp),
                "mean_start_temp": np.mean(start_temp),
                "sum_start_power": np.sum(start_power),
                "max_start_power": np.max(start_power),
                "num_active_cores": np.sum(active),
                "max_new_freq": np.max(new_f),
                "sum_new_freq": np.sum(new_f),
                "max_delta_freq": np.max(delta_f),
                "sum_delta_freq": np.sum(delta_f),
            }
            for k, v in globals_dict.items():
                features.append(float(v))
                names.append(k)

            # Per-core features. Only the ones with direct physical meaning.
            per_core_vectors = {
                "old_freq": old_f,
                "new_freq": new_f,
                "delta_freq": delta_f,
                "active": active,
                "start_temp": start_temp,
                "neighbor_temp": temp_neighbor,
                "start_power": start_power,
                "neighbor_power": power_neighbor,
                "start_util": start_util,
                "start_ipc": start_ipc,
                "start_rel_nuca_cpi": start_rel_nuca_cpi,
                "log_start_ips": log_start_ips,
                "workload_index": workload_index,
            }
            for base_name, vec in per_core_vectors.items():
                for i, v in enumerate(vec):
                    features.append(float(v))
                    names.append(f"{base_name}_c{i}")

            # Phase as a single integer categorical feature.
            features.append(float(PHASE_TO_INT[phase]))
            names.append("phase_id")

            X = np.array(features, dtype=np.float32)
            Y = delta_temp.astype(np.float32)

            X_rows.append(X)
            Y_rows.append(Y)
            start_temp_rows.append(start_temp.astype(np.float32))
            phase_rows.append(phase)
            kept_indices.append(idx)

            if feature_names is None:
                feature_names = names

        except Exception:
            skipped_parse_error += 1
            continue

    if not X_rows:
        raise ValueError("No valid rows left after cleaning. Check input CSV.")

    cleaned_df = raw_df.loc[kept_indices].copy()
    cleaned_df.insert(0, "source_row", kept_indices)
    cleaned_df = cleaned_df.reset_index(drop=True)

    target_names = [f"delta_temp_c{i}" for i in range(NUM_CORES)]

    # The categorical feature index is the last column (phase_id).
    categorical_feature_indices = [len(feature_names) - 1]

    print("\nData cleaning")
    print("-------------")
    print(f"raw rows              : {len(raw_df)}")
    print(f"kept rows             : {len(cleaned_df)}")
    print(f"removed temp -1 rows  : {skipped_invalid_temp}")
    print(f"removed invalid IPS rows: {skipped_invalid_ips}")
    print(f"removed transition rows: {skipped_transition_phase}")
    print(f"removed parse errors  : {skipped_parse_error}")

    phase_counts = pd.Series(phase_rows).value_counts()
    print("\nPhase counts")
    print("------------")
    for phase_name in PHASE_NAMES:
        print(f"{phase_name:15s}: {int(phase_counts.get(phase_name, 0))}")

    return DatasetBundle(
        X=np.stack(X_rows),
        Y=np.stack(Y_rows),
        start_temps=np.stack(start_temp_rows),
        phases=np.array(phase_rows),
        feature_names=feature_names,
        target_names=target_names,
        categorical_feature_indices=categorical_feature_indices,
        df=cleaned_df,
    )


# ------------------------------------------------------------
# Sample weighting
# ------------------------------------------------------------

def make_sample_weights(start_temps, Y, args):
    """
    Boost samples where the true end temperature is high, so the model
    stops systematically underestimating peaks.
    """
    end_temps = start_temps + Y
    peak_per_row = np.max(end_temps, axis=1)

    weights = np.ones(len(peak_per_row), dtype=np.float64)
    weights[peak_per_row > args.weight_t1] = args.weight_w1
    weights[peak_per_row > args.weight_t2] = args.weight_w2
    weights[peak_per_row > args.weight_t3] = args.weight_w3
    return weights


# ------------------------------------------------------------
# Metrics (identical to Ridge version)
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(Y_true, Y_pred, start_temps):
    true_end_temp = start_temps + Y_true
    pred_end_temp = start_temps + Y_pred

    temp_errors = pred_end_temp.reshape(-1) - true_end_temp.reshape(-1)
    abs_temp_errors = np.abs(temp_errors)

    temp_rmse = rmse(true_end_temp.reshape(-1), pred_end_temp.reshape(-1))
    temp_mae = mean_absolute_error(true_end_temp.reshape(-1), pred_end_temp.reshape(-1))
    temp_r2 = r2_score(true_end_temp.reshape(-1), pred_end_temp.reshape(-1))

    temp_within_3 = np.mean(abs_temp_errors <= 3.0)
    temp_within_5 = np.mean(abs_temp_errors <= 5.0)
    temp_within_10 = np.mean(abs_temp_errors <= 10.0)

    true_peak = np.max(true_end_temp, axis=1)
    pred_peak = np.max(pred_end_temp, axis=1)

    peak_rmse = rmse(true_peak, pred_peak)
    peak_mae = mean_absolute_error(true_peak, pred_peak)
    peak_r2 = r2_score(true_peak, pred_peak)

    return {
        "temp_rmse_c": temp_rmse,
        "temp_mae_c": temp_mae,
        "temp_r2": temp_r2,
        "temp_within_3c": temp_within_3,
        "temp_within_5c": temp_within_5,
        "temp_within_10c": temp_within_10,
        "peak_rmse_c": peak_rmse,
        "peak_mae_c": peak_mae,
        "peak_r2": peak_r2,
    }


def print_metric_table(title, metrics_list):
    print("\n" + title)
    print("-" * len(title))
    keys = metrics_list[0].keys()
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float64)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        print(f"{key:28s} mean={mean:10.4f}   std={std:10.4f}")


def print_metrics_by_phase(title, Y_true, Y_pred, start_temps, phases):
    print("\n" + title)
    print("-" * len(title))
    for phase_name in PHASE_NAMES:
        idx = np.where(phases == phase_name)[0]
        if len(idx) == 0:
            continue
        m = compute_metrics(Y_true[idx], Y_pred[idx], start_temps[idx])
        print(
            f"{phase_name:15s} "
            f"n={len(idx):5d}  "
            f"temp_mae={m['temp_mae_c']:8.4f}  "
            f"temp_rmse={m['temp_rmse_c']:8.4f}  "
            f"peak_mae={m['peak_mae_c']:8.4f}  "
            f"within_5c={m['temp_within_5c']:6.3f}"
        )


def print_metrics_by_temp_band(title, Y_true, Y_pred, start_temps):
    """
    Critical for this project: how well does the model do in the dangerous
    high-temperature region? Average MAE hides catastrophic failures.
    """
    true_end_temp = start_temps + Y_true
    pred_end_temp = start_temps + Y_pred

    true_flat = true_end_temp.reshape(-1)
    pred_flat = pred_end_temp.reshape(-1)

    bands = [
        ("< 60 C   ", true_flat < 60),
        ("60-80 C  ", (true_flat >= 60) & (true_flat < 80)),
        ("80-95 C  ", (true_flat >= 80) & (true_flat < 95)),
        ("95-110 C ", (true_flat >= 95) & (true_flat < 110)),
        (">= 110 C ", true_flat >= 110),
    ]

    print("\n" + title)
    print("-" * len(title))
    print(f"{'band':12s} {'n':>6s}  {'mae':>8s}  {'rmse':>8s}  {'bias':>8s}")
    for label, mask in bands:
        n = int(np.sum(mask))
        if n == 0:
            print(f"{label:12s} {n:>6d}  {'-':>8s}  {'-':>8s}  {'-':>8s}")
            continue
        err = pred_flat[mask] - true_flat[mask]
        mae = np.mean(np.abs(err))
        rms = math.sqrt(np.mean(err ** 2))
        bias = np.mean(err)  # negative bias = systematic underestimation
        print(f"{label:12s} {n:>6d}  {mae:>8.3f}  {rms:>8.3f}  {bias:>8.3f}")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

def make_core_model(args):
    """
    One LightGBM regressor per core. Returns an unfitted estimator.
    """
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_per_core_models(X_train, Y_train, sample_weight, categorical_indices, args):
    """
    Train one LightGBM model per output (per core).
    """
    models = []
    for core_idx in range(NUM_CORES):
        m = make_core_model(args)
        m.fit(
            X_train,
            Y_train[:, core_idx],
            sample_weight=sample_weight,
            categorical_feature=categorical_indices,
        )
        models.append(m)
    return models


def predict_per_core_models(models, X):
    # Use the underlying Booster directly to bypass sklearn's feature-name
    # check (we pass plain numpy arrays everywhere).
    preds = np.stack([m.booster_.predict(X) for m in models], axis=1)
    return preds.astype(np.float32)


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------

def train_and_predict_fold(X, Y, start_temps, phases, train_idx, test_idx,
                            categorical_indices, args):
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    start_train = start_temps[train_idx]
    start_test = start_temps[test_idx]
    phase_train = phases[train_idx]
    phase_test = phases[test_idx]

    sw = make_sample_weights(start_train, Y_train, args)

    models = fit_per_core_models(X_train, Y_train, sw, categorical_indices, args)

    Y_train_pred = predict_per_core_models(models, X_train)
    Y_test_pred = predict_per_core_models(models, X_test)

    train_metrics = compute_metrics(Y_train, Y_train_pred, start_train)
    test_metrics = compute_metrics(Y_test, Y_test_pred, start_test)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "models": models,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_train_pred": Y_train_pred,
        "Y_test_pred": Y_test_pred,
        "start_train": start_train,
        "start_test": start_test,
        "phase_train": phase_train,
        "phase_test": phase_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def run_cross_validation(bundle, args):
    X = bundle.X
    Y = bundle.Y
    start_temps = bundle.start_temps
    phases = bundle.phases

    n_samples = len(X)
    if n_samples < 5:
        raise ValueError("Too few samples for cross-validation. Need at least 5 rows.")

    folds = min(args.folds, n_samples)
    kf = KFold(n_splits=folds, shuffle=True, random_state=args.seed)

    train_metrics_all = []
    test_metrics_all = []

    all_test_y = []
    all_test_pred = []
    all_test_start = []
    all_test_phase = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold_id}/{folds}")
        print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")

        result = train_and_predict_fold(
            X, Y, start_temps, phases,
            train_idx, test_idx,
            bundle.categorical_feature_indices,
            args,
        )

        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]

        train_metrics_all.append(train_metrics)
        test_metrics_all.append(test_metrics)

        all_test_y.append(result["Y_test"])
        all_test_pred.append(result["Y_test_pred"])
        all_test_start.append(result["start_test"])
        all_test_phase.append(result["phase_test"])

        print(f"test temp_rmse_c : {test_metrics['temp_rmse_c']:.4f}")
        print(f"test temp_mae_c  : {test_metrics['temp_mae_c']:.4f}")
        print(f"test temp_r2     : {test_metrics['temp_r2']:.4f}")
        print(f"test peak_rmse_c : {test_metrics['peak_rmse_c']:.4f}")
        print(f"test peak_mae_c  : {test_metrics['peak_mae_c']:.4f}")

    print_metric_table("TRAIN metrics across folds", train_metrics_all)
    print_metric_table("TEST / CV metrics across folds", test_metrics_all)

    all_test_y = np.concatenate(all_test_y, axis=0)
    all_test_pred = np.concatenate(all_test_pred, axis=0)
    all_test_start = np.concatenate(all_test_start, axis=0)
    all_test_phase = np.concatenate(all_test_phase, axis=0)

    print_metrics_by_phase(
        "Combined TEST metrics by phase",
        all_test_y, all_test_pred, all_test_start, all_test_phase,
    )

    print_metrics_by_temp_band(
        "Combined TEST metrics by true-end-temp band",
        all_test_y, all_test_pred, all_test_start,
    )

    return train_metrics_all, test_metrics_all


def train_final_model(bundle, args):
    X = bundle.X
    Y = bundle.Y

    sw = make_sample_weights(bundle.start_temps, Y, args)

    models = fit_per_core_models(X, Y, sw, bundle.categorical_feature_indices, args)

    Y_pred = predict_per_core_models(models, X)

    train_all_metrics = compute_metrics(Y, Y_pred, bundle.start_temps)

    saved = {
        "models": models,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "phase_names": PHASE_NAMES,
        "phase_to_int": PHASE_TO_INT,
        "categorical_feature_indices": bundle.categorical_feature_indices,
        "model_type": "lightgbm",
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
    }

    joblib.dump(saved, args.output)

    cpp_output = args.cpp_output
    if cpp_output is None:
        output_root, _ = os.path.splitext(args.output)
        cpp_output = output_root + ".txt"

    export_cpp_model(saved, cpp_output)

    print("\nFINAL model trained on all cleaned data")
    print("---------------------------------------")
    for k, v in train_all_metrics.items():
        print(f"{k:28s} {v:10.4f}")

    print_metrics_by_phase(
        "FINAL train-all metrics by phase",
        Y, Y_pred, bundle.start_temps, bundle.phases,
    )
    print_metrics_by_temp_band(
        "FINAL train-all metrics by true-end-temp band",
        Y, Y_pred, bundle.start_temps,
    )

    # Feature importance (averaged across the 4 per-core models).
    print_feature_importance(models, bundle.feature_names, top_k=20)

    print(f"\nSaved model to: {args.output}")
    print(f"Saved C++ model to: {cpp_output}")

    return saved, train_all_metrics


def print_feature_importance(models, feature_names, top_k=20):
    """
    Average gain-based importance across all per-core models.
    Useful for trimming features in a future iteration.
    """
    importances = np.zeros(len(feature_names), dtype=np.float64)
    for m in models:
        booster = m.booster_
        imp = booster.feature_importance(importance_type="gain")
        importances += imp
    importances /= len(models)

    order = np.argsort(-importances)
    print(f"\nTop {top_k} features by average gain across cores")
    print("-" * 48)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:3d}. {feature_names[idx]:35s}  gain={importances[idx]:12.1f}")


# ------------------------------------------------------------
# C++ export
# ------------------------------------------------------------

def dump_tree_to_dict(tree_dict):
    """
    LightGBM dumps a tree as a nested dict.
    We flatten it into arrays so C++ can traverse with one array per field.

    For each node we emit:
        split_feature : feature index for internal nodes, -1 for leaves
        threshold     : split threshold for internal nodes, leaf value for leaves
        left_child    : index of left child, or -1
        right_child   : index of right child, or -1
        decision_type : 'le' for numeric <=, 'eq' for categorical ==
        cat_boundary  : if categorical, the categorical value used (single int)
    """
    nodes = []

    def add_node(node):
        node_idx = len(nodes)
        nodes.append(None)  # placeholder so children can be appended after

        if "leaf_value" in node:
            entry = {
                "is_leaf": True,
                "leaf_value": float(node["leaf_value"]),
                "split_feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "decision_type": "leaf",
                "cat_values": [],
            }
            nodes[node_idx] = entry
            return node_idx

        decision_type = node.get("decision_type", "<=")
        split_feature = int(node["split_feature"])

        if decision_type == "==":
            # Categorical split. LightGBM stores the matching category set.
            # For our single-int categorical (phase_id) it is usually a
            # single value or a small list of ints.
            threshold_str = node["threshold"]
            if isinstance(threshold_str, str):
                cat_values = [int(x) for x in threshold_str.split("||")]
            else:
                cat_values = [int(threshold_str)]
            entry = {
                "is_leaf": False,
                "leaf_value": 0.0,
                "split_feature": split_feature,
                "threshold": 0.0,
                "left": -1,  # filled in after recursion
                "right": -1,
                "decision_type": "eq",
                "cat_values": cat_values,
            }
        else:
            entry = {
                "is_leaf": False,
                "leaf_value": 0.0,
                "split_feature": split_feature,
                "threshold": float(node["threshold"]),
                "left": -1,
                "right": -1,
                "decision_type": "le",
                "cat_values": [],
            }

        nodes[node_idx] = entry

        left_idx = add_node(node["left_child"])
        right_idx = add_node(node["right_child"])
        entry["left"] = left_idx
        entry["right"] = right_idx

        return node_idx

    add_node(tree_dict["tree_structure"])
    return nodes


def export_cpp_model(saved, output_path):
    """
    Plain-text dump of all trees. The C++ side just needs:

        for each core c in 0..3:
            sum = 0
            for each tree t in trees[c]:
                node = 0
                while not leaf:
                    if categorical:
                        node = (x[split_feature] in cat_values) ? left : right
                    else:
                        node = (x[split_feature] <= threshold) ? left : right
                sum += leaf_value
            delta_temp_c = sum
    """
    models = saved["models"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]
    phase_names = saved["phase_names"]
    cat_indices = saved["categorical_feature_indices"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("THERMAL_LGBM_MODEL_V1\n")
        f.write(f"num_cores {NUM_CORES}\n")
        f.write(f"num_features {len(feature_names)}\n")
        f.write(f"num_targets {len(target_names)}\n")

        f.write("phase_names\n")
        for name in phase_names:
            f.write(f"{name}\n")

        f.write("feature_names\n")
        for name in feature_names:
            f.write(f"{name}\n")

        f.write("target_names\n")
        for name in target_names:
            f.write(f"{name}\n")

        f.write("categorical_feature_indices\n")
        f.write(" ".join(str(i) for i in cat_indices) + "\n")

        for core_idx, model in enumerate(models):
            booster = model.booster_
            dump = booster.dump_model()
            trees = dump["tree_info"]
            f.write(f"core {core_idx} num_trees {len(trees)}\n")

            for tree_id, tree in enumerate(trees):
                nodes = dump_tree_to_dict(tree)
                f.write(f"tree {tree_id} num_nodes {len(nodes)}\n")
                for node in nodes:
                    if node["is_leaf"]:
                        f.write(f"L {node['leaf_value']:.17g}\n")
                    elif node["decision_type"] == "eq":
                        cat_str = ",".join(str(c) for c in node["cat_values"])
                        f.write(
                            f"C {node['split_feature']} {cat_str} "
                            f"{node['left']} {node['right']}\n"
                        )
                    else:
                        f.write(
                            f"N {node['split_feature']} {node['threshold']:.17g} "
                            f"{node['left']} {node['right']}\n"
                        )

    # Also dump a JSON for easier debugging / sanity checking from any language.
    json_path = os.path.splitext(output_path)[0] + ".json"
    json_dump = {
        "model_format": "THERMAL_LGBM_MODEL_V1",
        "num_cores": NUM_CORES,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "target_names": target_names,
        "phase_names": phase_names,
        "categorical_feature_indices": cat_indices,
        "trees_per_core": [
            [dump_tree_to_dict(t) for t in m.booster_.dump_model()["tree_info"]]
            for m in models
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dump, f)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        default="ml_models/lgbm_thermal_model.joblib",
    )
    parser.add_argument(
        "--cpp-output",
        default=None,
        help="Where to save the C++-readable text model. Default: output path with .txt suffix",
    )

    # ---- LightGBM hyperparameters ----
    # The 5 hyperparams below are the ones you should actually tune.
    parser.add_argument(
        "--n-estimators", type=int, default=300,
        help="Number of boosting rounds (trees). 200-500 for ~500 samples.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="Smaller = more conservative; pair with more estimators.",
    )
    parser.add_argument(
        "--num-leaves", type=int, default=15,
        help="Max leaves per tree. Most important capacity knob. "
             "Small data (<1k rows): 8-31.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Max depth per tree. -1 = no limit. Pair with num_leaves.",
    )
    parser.add_argument(
        "--min-child-samples", type=int, default=10,
        help="Min samples per leaf. Small data: 5-20. Higher = less overfit.",
    )
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)

    # ---- Sample weighting for high-temp samples ----
    parser.add_argument("--weight-t1", type=float, default=70.0)
    parser.add_argument("--weight-w1", type=float, default=2.0)
    parser.add_argument("--weight-t2", type=float, default=85.0)
    parser.add_argument("--weight-w2", type=float, default=5.0)
    parser.add_argument("--weight-t3", type=float, default=100.0)
    parser.add_argument("--weight-w3", type=float, default=10.0)

    parser.add_argument("--print-features", action="store_true")

    args = parser.parse_args()

    bundle = build_dataset(args.csv)

    print("\nLoaded dataset")
    print("--------------")
    print(f"rows / samples : {bundle.X.shape[0]}")
    print(f"input features : {bundle.X.shape[1]}")
    print(f"targets        : {bundle.Y.shape[1]}")
    print(f"categorical idx: {bundle.categorical_feature_indices}")
    print(f"\nLightGBM hyperparameters")
    print(f"  n_estimators       : {args.n_estimators}")
    print(f"  learning_rate      : {args.learning_rate}")
    print(f"  num_leaves         : {args.num_leaves}")
    print(f"  max_depth          : {args.max_depth}")
    print(f"  min_child_samples  : {args.min_child_samples}")
    print(f"  reg_lambda         : {args.reg_lambda}")
    print(f"  subsample          : {args.subsample}")
    print(f"  colsample_bytree   : {args.colsample_bytree}")
    print(f"\nSample weighting")
    print(f"  peak > {args.weight_t1:5.1f} C  -> weight {args.weight_w1}")
    print(f"  peak > {args.weight_t2:5.1f} C  -> weight {args.weight_w2}")
    print(f"  peak > {args.weight_t3:5.1f} C  -> weight {args.weight_w3}")

    if args.print_features:
        print("\nFeature names:")
        for i, name in enumerate(bundle.feature_names):
            print(f"{i:03d}: {name}")
        return

    run_cross_validation(bundle, args)
    train_final_model(bundle, args)


if __name__ == "__main__":
    main()