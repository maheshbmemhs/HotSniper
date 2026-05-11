import argparse
import math
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

NUM_CORES = 4

PHASE_NAMES = [
    "master_only",
    "workers_only",
]


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


def add_vector_features(feature_list, name_list, base_name, vec):
    for i, v in enumerate(vec):
        feature_list.append(float(v))
        name_list.append(f"{base_name}_c{i}")


def classify_phase(start_util):
    """
    Two phase classes from per-core utilization.

    Returns None for transition intervals because the C++ scheduler bypasses
    ML prediction and pins all cores to 2GHz in that state.

    master_only:
        no worker core is active.

    workers_only:
        at least one worker core is active.
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


def add_phase_features(features, names, phase, use_interactions=True):
    """
    Add phase one-hot features and optional phase-feature interactions.

    This still trains only one Ridge model, but lets the linear model learn
    different slopes for different execution phases.
    """
    base_features = list(features)
    base_names = list(names)

    for phase_name in PHASE_NAMES:
        flag = 1.0 if phase == phase_name else 0.0
        features.append(flag)
        names.append(f"phase_{phase_name}")

    if use_interactions:
        for phase_name in PHASE_NAMES:
            flag = 1.0 if phase == phase_name else 0.0
            for fname, fval in zip(base_names, base_features):
                features.append(flag * float(fval))
                names.append(f"{phase_name}_x_{fname}")


def row_has_invalid_temperature(row):
    """
    Remove startup / corrupted rows where temperatures are -1.
    """
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
    df: pd.DataFrame


def build_dataset(csv_path, use_phase_interactions=True):
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

            # Frequencies in GHz
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

            temp_centered = start_temp - 55.0
            temp_to_peak = start_peak - start_temp
            temp_neighbor = neighbor_mean(start_temp)
            power_neighbor = neighbor_mean(start_power)

            new_f2 = new_f ** 2
            new_f3 = new_f ** 3
            old_f2 = old_f ** 2
            old_f3 = old_f ** 3

            active_new_f = active * new_f
            active_new_f3 = active * new_f3

            util_new_f = start_util * new_f
            util_new_f3 = start_util * new_f3

            workload_index = active * start_util * new_f3 * start_ipc

            temp_power_interaction = temp_centered * start_power
            temp_freq_interaction = temp_centered * new_f
            temp_work_interaction = temp_centered * workload_index

            features = []
            names = []

            global_features = {
                "start_peak_temp": start_peak,
                "mean_start_temp": np.mean(start_temp),
                "max_start_temp": np.max(start_temp),
                "min_start_temp": np.min(start_temp),
                "std_start_temp": np.std(start_temp),

                "mean_start_power": np.mean(start_power),
                "sum_start_power": np.sum(start_power),
                "max_start_power": np.max(start_power),

                "mean_start_util": np.mean(start_util),
                "sum_start_util": np.sum(start_util),

                "num_active_cores": np.sum(active),

                "mean_old_freq": np.mean(old_f),
                "mean_new_freq": np.mean(new_f),
                "max_new_freq": np.max(new_f),
                "min_new_freq": np.min(new_f),
                "sum_new_freq": np.sum(new_f),

                "mean_delta_freq": np.mean(delta_f),
                "max_delta_freq": np.max(delta_f),
                "min_delta_freq": np.min(delta_f),
                "sum_delta_freq": np.sum(delta_f),
            }

            for k, v in global_features.items():
                features.append(float(v))
                names.append(k)

            per_core_vectors = {
                "old_freq": old_f,
                "new_freq": new_f,
                "delta_freq": delta_f,

                "new_freq_sq": new_f2,
                "new_freq_cube": new_f3,
                "old_freq_sq": old_f2,
                "old_freq_cube": old_f3,

                "active": active,
                "active_new_freq": active_new_f,
                "active_new_freq_cube": active_new_f3,

                "start_temp": start_temp,
                "temp_centered": temp_centered,
                "temp_to_peak": temp_to_peak,
                "neighbor_temp": temp_neighbor,

                "start_power": start_power,
                "neighbor_power": power_neighbor,

                "start_util": start_util,
                "util_new_freq": util_new_f,
                "util_new_freq_cube": util_new_f3,

                "start_ipc": start_ipc,
                "start_rel_nuca_cpi": start_rel_nuca_cpi,

                "log_start_ips": log_start_ips,

                "workload_index": workload_index,
                "temp_power_interaction": temp_power_interaction,
                "temp_freq_interaction": temp_freq_interaction,
                "temp_work_interaction": temp_work_interaction,
            }

            for base_name, vec in per_core_vectors.items():
                add_vector_features(features, names, base_name, vec)

            add_phase_features(
                features,
                names,
                phase,
                use_interactions=use_phase_interactions,
            )

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
        df=cleaned_df,
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(Y_true, Y_pred, start_temps):
    true_delta_temp = Y_true
    pred_delta_temp = Y_pred

    true_end_temp = start_temps + true_delta_temp
    pred_end_temp = start_temps + pred_delta_temp

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


def format_vec(values):
    return ";".join(f"{float(v):.3f}" for v in values)


def print_large_error_rows(
    title,
    df,
    Y_true,
    Y_pred,
    start_temps,
    phases,
    threshold=10.0,
    max_rows=50,
):
    true_end_temp = start_temps + Y_true
    pred_end_temp = start_temps + Y_pred
    errors = pred_end_temp - true_end_temp
    abs_errors = np.abs(errors)
    row_mask = np.any(abs_errors > threshold, axis=1)
    bad_indices = np.where(row_mask)[0]

    print("\n" + title)
    print("-" * len(title))
    print(f"threshold C: {threshold:.3f}")
    print(f"rows over threshold: {len(bad_indices)} / {len(df)}")

    if len(bad_indices) == 0:
        return

    if max_rows is not None and max_rows > 0:
        shown_indices = bad_indices[:max_rows]
    else:
        shown_indices = bad_indices

    for local_idx in shown_indices:
        row = df.iloc[int(local_idx)]
        row_abs_errors = abs_errors[local_idx]
        worst_core = int(np.argmax(row_abs_errors))

        prefix = (
            f"dataset_idx={int(local_idx)} "
            f"source_row={int(row['source_row']) if 'source_row' in row else int(local_idx)} "
            f"phase={phases[local_idx]} "
            f"worst_core=c{worst_core} "
            f"max_abs_error_c={float(row_abs_errors[worst_core]):.3f}"
        )

        if "experiment" in row:
            prefix += f" experiment={row['experiment']}"
        if "cycle" in row:
            prefix += f" cycle={row['cycle']}"

        print(prefix)
        print(f"  true_end_temp_c : {format_vec(true_end_temp[local_idx])}")
        print(f"  pred_end_temp_c : {format_vec(pred_end_temp[local_idx])}")
        print(f"  abs_error_c     : {format_vec(row_abs_errors)}")

        for col in (
            "old_frequencies_mhz",
            "new_frequencies_mhz",
            "active_cores",
            "start_core_utilizations",
            "start_core_ips",
        ):
            if col in row:
                print(f"  {col}: {row[col]}")

    remaining = len(bad_indices) - len(shown_indices)
    if remaining > 0:
        print(f"... {remaining} more rows omitted; use --max-error-rows 0 to print all")


# ------------------------------------------------------------
# Ridge model
# ------------------------------------------------------------

def make_model(args):
    return Ridge(
        alpha=args.ridge_alpha,
        fit_intercept=True,
        random_state=args.seed,
    )


def get_model_description(model):
    info = {
        "model_class": model.__class__.__name__,
    }

    if hasattr(model, "alpha"):
        info["alpha"] = model.alpha

    return info


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------

def train_and_predict_fold(X, Y, start_temps, phases, train_idx, test_idx, args):
    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    start_train = start_temps[train_idx]
    start_test = start_temps[test_idx]

    phase_train = phases[train_idx]
    phase_test = phases[test_idx]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    Y_train_s = y_scaler.fit_transform(Y_train)

    model = make_model(args)
    model.fit(X_train_s, Y_train_s)

    Y_train_pred_s = model.predict(X_train_s)
    Y_test_pred_s = model.predict(X_test_s)

    Y_train_pred = y_scaler.inverse_transform(Y_train_pred_s)
    Y_test_pred = y_scaler.inverse_transform(Y_test_pred_s)

    train_metrics = compute_metrics(Y_train, Y_train_pred, start_train)
    test_metrics = compute_metrics(Y_test, Y_test_pred, start_test)

    model_info = get_model_description(model)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_info": model_info,
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
    model_info_all = []

    all_test_y = []
    all_test_pred = []
    all_test_start = []
    all_test_phase = []
    all_test_indices = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold_id}/{folds}")
        print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")

        result = train_and_predict_fold(
            X,
            Y,
            start_temps,
            phases,
            train_idx,
            test_idx,
            args,
        )

        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]
        model_info = result["model_info"]

        train_metrics_all.append(train_metrics)
        test_metrics_all.append(test_metrics)
        model_info_all.append(model_info)

        all_test_y.append(result["Y_test"])
        all_test_pred.append(result["Y_test_pred"])
        all_test_start.append(result["start_test"])
        all_test_phase.append(result["phase_test"])
        all_test_indices.append(result["test_idx"])

        print("model info       :", model_info)
        print("test temp_rmse_c :", f"{test_metrics['temp_rmse_c']:.4f}")
        print("test temp_mae_c  :", f"{test_metrics['temp_mae_c']:.4f}")
        print("test temp_r2     :", f"{test_metrics['temp_r2']:.4f}")
        print("test peak_rmse_c :", f"{test_metrics['peak_rmse_c']:.4f}")

        if args.print_phase_metrics_each_fold:
            print_metrics_by_phase(
                f"Fold {fold_id} TEST metrics by phase",
                result["Y_test"],
                result["Y_test_pred"],
                result["start_test"],
                result["phase_test"],
            )

    print_metric_table("TRAIN metrics across folds", train_metrics_all)
    print_metric_table("TEST / CV metrics across folds", test_metrics_all)

    all_test_y = np.concatenate(all_test_y, axis=0)
    all_test_pred = np.concatenate(all_test_pred, axis=0)
    all_test_start = np.concatenate(all_test_start, axis=0)
    all_test_phase = np.concatenate(all_test_phase, axis=0)
    all_test_indices = np.concatenate(all_test_indices, axis=0)

    print_metrics_by_phase(
        "Combined TEST metrics by phase",
        all_test_y,
        all_test_pred,
        all_test_start,
        all_test_phase,
    )

    # print_large_error_rows(
    #     "Combined TEST rows with > threshold end-temp error",
    #     bundle.df.iloc[all_test_indices].reset_index(drop=True),
    #     all_test_y,
    #     all_test_pred,
    #     all_test_start,
    #     all_test_phase,
    #     threshold=args.error_threshold,
    #     max_rows=args.max_error_rows,
    # )

    return train_metrics_all, test_metrics_all, model_info_all


def train_final_model(bundle, args):
    X = bundle.X
    Y = bundle.Y

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_s = x_scaler.fit_transform(X)
    Y_s = y_scaler.fit_transform(Y)

    model = make_model(args)
    model.fit(X_s, Y_s)

    Y_pred_s = model.predict(X_s)
    Y_pred = y_scaler.inverse_transform(Y_pred_s)

    train_all_metrics = compute_metrics(Y, Y_pred, bundle.start_temps)
    model_info = get_model_description(model)

    saved = {
        "model": model,
        "model_info": model_info,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "phase_names": PHASE_NAMES,
        "model_type": args.model,
        "uses_phase_interactions": not args.no_phase_interactions,
    }

    joblib.dump(saved, args.output)

    cpp_output = args.cpp_output
    if cpp_output is None:
        output_root, _ = os.path.splitext(args.output)
        cpp_output = output_root + ".txt"

    export_cpp_model(saved, cpp_output)

    print("\nFINAL model trained on all cleaned data")
    print("---------------------------------------")
    print("model info:", model_info)

    for k, v in train_all_metrics.items():
        print(f"{k:28s} {v:10.4f}")

    print_metrics_by_phase(
        "FINAL train-all metrics by phase",
        Y,
        Y_pred,
        bundle.start_temps,
        bundle.phases,
    )

    print_large_error_rows(
        "FINAL train-all rows with > threshold end-temp error",
        bundle.df,
        Y,
        Y_pred,
        bundle.start_temps,
        bundle.phases,
        threshold=args.error_threshold,
        max_rows=args.max_error_rows,
    )

    print(f"\nSaved model to: {args.output}")
    print(f"Saved C++ model to: {cpp_output}")

    return saved, train_all_metrics


# ------------------------------------------------------------
# C++ export
# ------------------------------------------------------------

def write_vector(file, values):
    file.write(" ".join(f"{float(v):.17g}" for v in values))
    file.write("\n")


def export_cpp_model(saved, output_path):
    model = saved["model"]
    x_scaler = saved["x_scaler"]
    y_scaler = saved["y_scaler"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]
    phase_names = saved.get("phase_names", PHASE_NAMES)

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("C++ export requires a fitted linear model with coef_ and intercept_")

    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    x_mean = np.asarray(x_scaler.mean_, dtype=np.float64)
    x_scale = np.asarray(x_scaler.scale_, dtype=np.float64)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

    if coef.shape != (len(target_names), len(feature_names)):
        raise ValueError(
            f"Unexpected coefficient shape {coef.shape}, expected "
            f"({len(target_names)}, {len(feature_names)})"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("THERMAL_REGRESSION_MODEL_V2_PHASE_AWARE\n")
        f.write("num_cores 4\n")
        f.write(f"num_features {len(feature_names)}\n")
        f.write(f"num_targets {len(target_names)}\n")
        f.write(f"uses_phase_interactions {int(saved.get('uses_phase_interactions', True))}\n")

        f.write("phase_names\n")
        for name in phase_names:
            f.write(f"{name}\n")

        f.write("feature_names\n")
        for name in feature_names:
            f.write(f"{name}\n")

        f.write("target_names\n")
        for name in target_names:
            f.write(f"{name}\n")

        f.write("x_mean\n")
        write_vector(f, x_mean)

        f.write("x_scale\n")
        write_vector(f, x_scale)

        f.write("y_mean\n")
        write_vector(f, y_mean)

        f.write("y_scale\n")
        write_vector(f, y_scale)

        f.write("coef\n")
        for row in coef:
            write_vector(f, row)

        f.write("intercept\n")
        write_vector(f, intercept)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default="ml_models/ridge_phase_aware_thermal_model.joblib",
        help="Where to save the final trained model",
    )

    parser.add_argument(
        "--cpp-output",
        default=None,
        help="Where to save the C++-readable text model. Default: output path with .txt suffix",
    )

    parser.add_argument(
        "--model",
        choices=["ridge"],
        default="ridge",
        help="Model type. Only ridge is currently supported.",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Alpha for Ridge. Phase interactions add many features, so 10.0 is a safer default than 1.0.",
    )

    parser.add_argument(
        "--no-phase-interactions",
        action="store_true",
        help="Disable phase x feature interactions. Phase one-hot features are still used.",
    )

    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print all feature names and exit after loading dataset",
    )

    parser.add_argument(
        "--print-phase-metrics-each-fold",
        action="store_true",
        help="Print per-phase metrics for every fold, not only the combined CV result.",
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=10.0,
        help="Print rows where any per-core end-temperature prediction error exceeds this threshold in C.",
    )

    parser.add_argument(
        "--max-error-rows",
        type=int,
        default=50,
        help="Maximum large-error rows to print per report. Use 0 to print all.",
    )

    args = parser.parse_args()

    bundle = build_dataset(
        args.csv,
        use_phase_interactions=not args.no_phase_interactions,
    )

    print("\nLoaded dataset")
    print("--------------")
    print(f"rows / samples: {bundle.X.shape[0]}")
    print(f"input features: {bundle.X.shape[1]}")
    print(f"targets       : {bundle.Y.shape[1]}")
    print("target names  :", bundle.target_names)
    print("model         :", args.model)
    print("ridge alpha   :", args.ridge_alpha)
    print("phase interactions:", not args.no_phase_interactions)

    if args.print_features:
        print("\nFeature names:")
        for i, name in enumerate(bundle.feature_names):
            print(f"{i:03d}: {name}")
        return

    run_cross_validation(bundle, args)
    train_final_model(bundle, args)


if __name__ == "__main__":
    main()
