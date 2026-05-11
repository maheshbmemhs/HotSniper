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


def parse_vec(value, expected_len=NUM_CORES, colname=""):
    """
    Convert a semicolon-separated string like:
        "1000;2000;3000;1000"
    into:
        np.array([1000, 2000, 3000, 1000], dtype=np.float32)
    """
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
    """
    IPS can be negative in startup artifacts, and inactive cores can have tiny IPS.
    We clip negative values to 0 and use log10(IPS + 1) as an input feature.
    """
    ips = np.asarray(ips, dtype=np.float32)
    ips = np.maximum(ips, 0.0)
    return np.log10(ips + 1.0)


def cpi_to_ipc(cpi):
    """
    Convert CPI to IPC.

    Inactive cores often have CPI like:
        1000000, 2000000, 3000000

    These are not meaningful CPI values, so we treat CPI >= 100 as invalid/inactive.
    """
    cpi = np.asarray(cpi, dtype=np.float32)
    ipc = np.zeros_like(cpi)

    valid = cpi < 100.0
    ipc[valid] = 1.0 / np.maximum(cpi[valid], 1e-6)

    return ipc


def neighbor_mean(x):
    """
    For 4 cores, return the average of the other 3 cores for each core.
    """
    x = np.asarray(x, dtype=np.float32)
    return (np.sum(x) - x) / 3.0


def add_vector_features(feature_list, name_list, base_name, vec):
    """
    Add per-core vector features into the global feature list.
    """
    for i, v in enumerate(vec):
        feature_list.append(float(v))
        name_list.append(f"{base_name}_c{i}")


# ------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------

@dataclass
class DatasetBundle:
    X: np.ndarray
    Y: np.ndarray
    start_temps: np.ndarray
    feature_names: list
    target_names: list
    df: pd.DataFrame


def build_dataset(csv_path):
    df = pd.read_csv(csv_path)

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

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    X_rows = []
    Y_rows = []
    start_temp_rows = []

    feature_names = None

    for _, row in df.iterrows():
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

        log_start_ips = safe_log_ips(start_ips)

        start_peak = float(row["start_peak_temp_c"])

        temp_centered = start_temp - 55.0
        temp_to_peak = start_peak - start_temp
        temp_neighbor = neighbor_mean(start_temp)

        power_neighbor = neighbor_mean(start_power)

        # Hand-crafted nonlinear features.
        # Even though the final model is linear, these engineered terms allow it to
        # model nonlinear physical relationships such as f^2, f^3, and interactions.
        new_f2 = new_f ** 2
        new_f3 = new_f ** 3

        old_f2 = old_f ** 2
        old_f3 = old_f ** 3

        active_new_f = active * new_f
        active_new_f3 = active * new_f3

        util_new_f = start_util * new_f
        util_new_f3 = start_util * new_f3

        # Workload heating index:
        # active × utilization × frequency^3 × IPC
        workload_index = active * start_util * new_f3 * start_ipc

        # Thermal interaction features
        temp_power_interaction = temp_centered * start_power
        temp_freq_interaction = temp_centered * new_f
        temp_work_interaction = temp_centered * workload_index

        # Global features
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

        # Per-core features
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

        X = np.array(features, dtype=np.float32)

        # Target: per-core temperature delta for the next scheduler interval.
        Y = delta_temp.astype(np.float32)

        X_rows.append(X)
        Y_rows.append(Y)
        start_temp_rows.append(start_temp.astype(np.float32))

        if feature_names is None:
            feature_names = names

    target_names = [f"delta_temp_c{i}" for i in range(NUM_CORES)]

    return DatasetBundle(
        X=np.stack(X_rows),
        Y=np.stack(Y_rows),
        start_temps=np.stack(start_temp_rows),
        feature_names=feature_names,
        target_names=target_names,
        df=df,
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(Y_true, Y_pred, start_temps):
    """
    Y format:
        [:, 0:4] = delta temp
    """
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

    # Peak temperature metrics
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
    """
    Print mean and std across folds.
    """
    print("\n" + title)
    print("-" * len(title))

    keys = metrics_list[0].keys()

    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float64)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        print(f"{key:28s} mean={mean:10.4f}   std={std:10.4f}")


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
    """
    Return a small dict describing the fitted model.
    """
    info = {
        "model_class": model.__class__.__name__,
    }

    if hasattr(model, "alpha"):
        info["alpha"] = model.alpha

    return info


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------

def train_and_predict_fold(X, Y, start_temps, train_idx, test_idx, args, fold_id):
    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    start_train = start_temps[train_idx]
    start_test = start_temps[test_idx]

    # Scale X and Y. X scaling is important for Ridge because features have
    # very different ranges. Y scaling keeps the per-core delta-temperature
    # targets numerically balanced.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    Y_train_s = y_scaler.fit_transform(Y_train)

    model = make_model(args)
    model.fit(X_train_s, Y_train_s)

    # Predict train and test
    Y_train_pred_s = model.predict(X_train_s)
    Y_test_pred_s = model.predict(X_test_s)

    Y_train_pred = y_scaler.inverse_transform(Y_train_pred_s)
    Y_test_pred = y_scaler.inverse_transform(Y_test_pred_s)

    train_metrics = compute_metrics(Y_train, Y_train_pred, start_train)
    test_metrics = compute_metrics(Y_test, Y_test_pred, start_test)

    model_info = get_model_description(model)

    return train_metrics, test_metrics, model_info


def run_cross_validation(bundle, args):
    X = bundle.X
    Y = bundle.Y
    start_temps = bundle.start_temps

    n_samples = len(X)

    if n_samples < 5:
        raise ValueError("Too few samples for cross-validation. Need at least 5 rows.")

    folds = min(args.folds, n_samples)

    kf = KFold(n_splits=folds, shuffle=True, random_state=args.seed)

    train_metrics_all = []
    test_metrics_all = []
    model_info_all = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold_id}/{folds}")
        print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")

        train_metrics, test_metrics, model_info = train_and_predict_fold(
            X,
            Y,
            start_temps,
            train_idx,
            test_idx,
            args,
            fold_id,
        )

        train_metrics_all.append(train_metrics)
        test_metrics_all.append(test_metrics)
        model_info_all.append(model_info)

        print("model info       :", model_info)
        print("test temp_rmse_c :", f"{test_metrics['temp_rmse_c']:.4f}")
        print("test temp_mae_c  :", f"{test_metrics['temp_mae_c']:.4f}")
        print("test temp_r2     :", f"{test_metrics['temp_r2']:.4f}")
        print("test peak_rmse_c :", f"{test_metrics['peak_rmse_c']:.4f}")

    print_metric_table("TRAIN metrics across folds", train_metrics_all)
    print_metric_table("TEST / CV metrics across folds", test_metrics_all)

    return train_metrics_all, test_metrics_all, model_info_all


def train_final_model(bundle, args):
    """
    Train final model on all samples and save it.
    Use this saved model later for prediction.
    """
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
        "model_type": args.model,
    }

    joblib.dump(saved, args.output)

    cpp_output = args.cpp_output
    if cpp_output is None:
        output_root, _ = os.path.splitext(args.output)
        cpp_output = output_root + ".txt"
    export_cpp_model(saved, cpp_output)

    print("\nFINAL model trained on all data")
    print("-------------------------------")
    print("model info:", model_info)

    for k, v in train_all_metrics.items():
        print(f"{k:28s} {v:10.4f}")

    print(f"\nSaved model to: {args.output}")
    print(f"Saved C++ model to: {cpp_output}")

    return saved, train_all_metrics


def write_vector(file, values):
    file.write(" ".join(f"{float(v):.17g}" for v in values))
    file.write("\n")


def export_cpp_model(saved, output_path):
    """
    Export the fitted linear/Ridge model in a plain text format that the C++
    scheduler can load without importing Python or sklearn.
    """
    model = saved["model"]
    x_scaler = saved["x_scaler"]
    y_scaler = saved["y_scaler"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]

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
        f.write("THERMAL_REGRESSION_MODEL_V1\n")
        f.write("num_cores 4\n")
        f.write(f"num_features {len(feature_names)}\n")
        f.write(f"num_targets {len(target_names)}\n")

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
        default="ml_models/ridge_thermal_ips_model.joblib",
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
        default=1.0,
        help="Alpha for --model ridge",
    )

    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print all feature names and exit after loading dataset",
    )

    args = parser.parse_args()

    bundle = build_dataset(args.csv)

    print("Loaded dataset")
    print("--------------")
    print(f"rows / samples: {bundle.X.shape[0]}")
    print(f"input features: {bundle.X.shape[1]}")
    print(f"targets       : {bundle.Y.shape[1]}")
    print("target names  :", bundle.target_names)
    print("model         :", args.model)
    print("ridge alpha   :", args.ridge_alpha)

    if args.print_features:
        print("\nFeature names:")
        for i, name in enumerate(bundle.feature_names):
            print(f"{i:03d}: {name}")
        return

    run_cross_validation(bundle, args)
    train_final_model(bundle, args)


if __name__ == "__main__":
    main()
