"""
Single-core LightGBM thermal delta predictor.

Input CSV format:
- Same interval-level CSV as before.
- Columns such as old_frequencies_mhz, start_core_temps_c, start_core_ips
  are semicolon-separated 4-core vectors.

Dataset transformation:
- Each original interval row is expanded into 4 target-core samples.
- Each sample predicts only one target core's temperature delta.
- The model is one shared LightGBM regressor, not four separate regressors.

Target:
    y = end_temp[target_core] - start_temp[target_core]

Prediction meaning:
    predicted_end_temp = start_temp[target_core] + model.predict(features)

Important:
- KFold is replaced by GroupKFold. The group is the original CSV row index,
  so core samples from the same interval never leak between train/test folds.
- Negative IPS artifacts are repaired from frequency and CPI when possible.
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
from sklearn.model_selection import GroupKFold


NUM_CORES = 4

PHASE_NAMES = [
    "master_only",
    "workers_only",
]

PHASE_TO_INT = {name: i for i, name in enumerate(PHASE_NAMES)}


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

def read_csv_auto(path):
    """
    Supports both comma CSV and tab-separated TSV.
    Your pasted example is tab-separated, but some generated files may be comma-separated.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    # Sometimes a TSV is accidentally read as one giant column.
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep="\t")

    return df


def parse_vec(value, expected_len=NUM_CORES, colname=""):
    if pd.isna(value):
        raise ValueError(f"Missing value in column {colname}")

    text = str(value).strip().strip('"').strip("'")
    parts = text.split(";")
    vals = np.array([float(x.strip()) for x in parts if x.strip() != ""], dtype=np.float32)

    if len(vals) != expected_len:
        raise ValueError(
            f"Column {colname} expected {expected_len} values, got {len(vals)}: {value}"
        )

    return vals


def active_thread_id_set(thread_ids):
    return sorted(int(x) for x in thread_ids if x >= 0)


def cpi_to_ipc(cpi):
    cpi = np.asarray(cpi, dtype=np.float32)
    ipc = np.zeros_like(cpi)

    valid = (cpi > 0.0) & (cpi < 100.0)
    ipc[valid] = 1.0 / np.maximum(cpi[valid], 1e-6)

    return ipc


def repair_ips_from_freq_cpi(raw_ips, freq_ghz, cpi):
    """
    HotSniper logs sometimes contain negative IPS artifacts, especially near cycle 0.
    Example from your pasted CSV:
        start_core_ips = -1901140.684 for core 0
    But CPI and frequency are valid, so we can reconstruct:
        IPS ~= frequency_hz / CPI
    """
    raw_ips = np.asarray(raw_ips, dtype=np.float32).copy()
    freq_ghz = np.asarray(freq_ghz, dtype=np.float32)
    cpi = np.asarray(cpi, dtype=np.float32)

    fixed = raw_ips.copy()

    for i in range(NUM_CORES):
        if fixed[i] >= 0.0:
            continue

        if cpi[i] > 0.0 and cpi[i] < 100.0 and freq_ghz[i] > 0.0:
            fixed[i] = (freq_ghz[i] * 1e9) / cpi[i]
        else:
            fixed[i] = 0.0

    return fixed


def estimate_power_for_candidate_freq(source_power, source_freq_ghz, target_freq_ghz):
    source_power = np.asarray(source_power, dtype=np.float32)
    source_freq_ghz = np.asarray(source_freq_ghz, dtype=np.float32)
    target_freq_ghz = np.asarray(target_freq_ghz, dtype=np.float32)

    ratio = np.ones_like(target_freq_ghz, dtype=np.float32)
    valid = source_freq_ghz > 0.0
    ratio[valid] = target_freq_ghz[valid] / np.maximum(source_freq_ghz[valid], 1e-6)
    return source_power * np.clip(ratio, 0.0, 4.0)


def infer_expected_runtime_from_thread_ids(
    old_f,
    new_f,
    active,
    start_power,
    start_util,
    start_cpi,
    start_ips,
    start_thread_ids=None,
    end_thread_ids=None,
):
    expected_active = active.copy()
    source_core = np.arange(NUM_CORES, dtype=np.int32)

    if start_thread_ids is not None and end_thread_ids is not None:
        expected_active = (end_thread_ids >= 0).astype(np.float32)
        for target_core in range(NUM_CORES):
            thread_id = end_thread_ids[target_core]
            if thread_id < 0:
                source_core[target_core] = target_core
                continue

            matches = np.where(start_thread_ids == thread_id)[0]
            source_core[target_core] = int(matches[0]) if len(matches) else target_core

    source_old_f = old_f[source_core]
    ratio = np.ones(NUM_CORES, dtype=np.float32)
    valid = source_old_f > 0.0
    ratio[valid] = new_f[valid] / np.maximum(source_old_f[valid], 1e-6)
    ratio = np.clip(ratio, 0.0, 4.0)

    expected_power = estimate_power_for_candidate_freq(
        source_power=start_power[source_core],
        source_freq_ghz=source_old_f,
        target_freq_ghz=new_f,
    )
    expected_util = start_util[source_core] * expected_active
    expected_cpi = start_cpi[source_core]
    expected_ips = start_ips[source_core] * ratio * expected_active

    return expected_active, expected_power, expected_util, expected_cpi, expected_ips


def safe_log_ips(ips):
    ips = np.asarray(ips, dtype=np.float32)
    ips = np.maximum(ips, 0.0)
    return np.log10(ips + 1.0)


def classify_phase(start_util):
    """
    Same phase logic as before:
    - Drop transition intervals if any core utilization is between 0.01 and 0.30.
    - If worker cores 1..3 are busy, classify as workers_only.
    - Otherwise classify as master_only.
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

    if float(row["end_peak_temp_c"]) < 0:
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
    end_temps: np.ndarray
    core_ids: np.ndarray
    phases: np.ndarray
    groups: np.ndarray
    feature_names: list
    target_name: str
    categorical_feature_indices: list
    df: pd.DataFrame


class FeatureBuilder:
    def __init__(self):
        self.features = []
        self.names = []

    def add(self, name, value):
        self.names.append(name)
        self.features.append(float(value))

    def to_array(self):
        return np.array(self.features, dtype=np.float32)


def build_single_core_features(
    target_core,
    start_peak,
    old_f,
    new_f,
    delta_f,
    active,
    expected_active,
    start_temp,
    start_power,
    expected_power,
    start_util,
    expected_util,
    start_ipc,
    expected_ipc,
    expected_cpi,
    start_rel_nuca_cpi,
    log_start_ips,
    log_expected_ips,
    phase,
):
    i = target_core
    other = [j for j in range(NUM_CORES) if j != i]

    busy_mask = start_util > 0.30
    expected_busy_mask = expected_util > 0.30

    workload_index = active[i] * start_util[i] * (new_f[i] ** 3) * start_ipc[i]
    expected_workload_index = active[i] * start_util[i] * (new_f[i] ** 3) * expected_ipc[i]
    expected_runtime_index = (
        expected_active[i] * expected_util[i] * (new_f[i] ** 3) * expected_ipc[i]
    )

    fb = FeatureBuilder()

    # ---- Global thermal / workload context ----
    fb.add("start_peak_temp", start_peak)
    fb.add("mean_start_temp", np.mean(start_temp))
    fb.add("max_start_temp", np.max(start_temp))
    fb.add("min_start_temp", np.min(start_temp))
    fb.add("temp_spread", np.max(start_temp) - np.min(start_temp))

    fb.add("sum_start_power", np.sum(start_power))
    fb.add("max_start_power", np.max(start_power))
    fb.add("mean_start_power", np.mean(start_power))
    fb.add("sum_expected_power", np.sum(expected_power))
    fb.add("max_expected_power", np.max(expected_power))
    fb.add("mean_expected_power", np.mean(expected_power))

    fb.add("num_active_flags", np.sum(active))
    fb.add("num_busy_cores", np.sum(busy_mask))
    fb.add("num_expected_active_flags", np.sum(expected_active))
    fb.add("num_expected_busy_cores", np.sum(expected_busy_mask))

    fb.add("max_new_freq", np.max(new_f))
    fb.add("mean_new_freq", np.mean(new_f))
    fb.add("sum_new_freq", np.sum(new_f))

    fb.add("max_delta_freq", np.max(delta_f))
    fb.add("min_delta_freq", np.min(delta_f))
    fb.add("sum_delta_freq", np.sum(delta_f))

    # ---- Target core identity and local features ----
    fb.add("core_id", i)

    fb.add("old_freq", old_f[i])
    fb.add("new_freq", new_f[i])
    fb.add("delta_freq", delta_f[i])

    fb.add("active", active[i])
    fb.add("is_busy", 1.0 if start_util[i] > 0.30 else 0.0)
    fb.add("expected_active", expected_active[i])
    fb.add("expected_active_delta", expected_active[i] - active[i])
    fb.add("expected_is_busy", 1.0 if expected_util[i] > 0.30 else 0.0)

    fb.add("start_temp", start_temp[i])
    fb.add("temp_vs_peak", start_temp[i] - start_peak)
    fb.add("temp_vs_mean", start_temp[i] - np.mean(start_temp))

    fb.add("start_power", start_power[i])
    fb.add("expected_power", expected_power[i])
    fb.add("expected_power_delta", expected_power[i] - start_power[i])
    fb.add("start_util", start_util[i])
    fb.add("expected_util", expected_util[i])
    fb.add("expected_util_delta", expected_util[i] - start_util[i])
    fb.add("start_ipc", start_ipc[i])
    fb.add("expected_ipc", expected_ipc[i])
    fb.add("expected_ipc_delta", expected_ipc[i] - start_ipc[i])
    fb.add("expected_cpi", expected_cpi[i])
    fb.add("start_rel_nuca_cpi", start_rel_nuca_cpi[i])
    fb.add("log_start_ips", log_start_ips[i])
    fb.add("log_expected_ips", log_expected_ips[i])

    fb.add("workload_index", workload_index)
    fb.add("expected_workload_index", expected_workload_index)
    fb.add("expected_runtime_index", expected_runtime_index)

    # ---- Neighbor features ----
    fb.add("neighbor_temp_mean", np.mean(start_temp[other]))
    fb.add("neighbor_temp_max", np.max(start_temp[other]))
    fb.add("neighbor_temp_min", np.min(start_temp[other]))
    fb.add("neighbor_power_mean", np.mean(start_power[other]))
    fb.add("neighbor_power_max", np.max(start_power[other]))
    fb.add("neighbor_expected_power_mean", np.mean(expected_power[other]))
    fb.add("neighbor_expected_power_max", np.max(expected_power[other]))
    fb.add("neighbor_util_mean", np.mean(start_util[other]))
    fb.add("neighbor_busy_count", np.sum(busy_mask[other]))
    fb.add("neighbor_expected_util_mean", np.mean(expected_util[other]))
    fb.add("neighbor_expected_busy_count", np.sum(expected_busy_mask[other]))

    # ---- Phase categorical feature ----
    fb.add("phase_id", PHASE_TO_INT[phase])

    return fb.to_array(), fb.names


def build_dataset(csv_path):
    raw_df = read_csv_auto(csv_path)

    required_cols = [
        "start_peak_temp_c",
        "end_peak_temp_c",
        "old_frequencies_mhz",
        "new_frequencies_mhz",
        "active_cores",
        "start_core_temps_c",
        "end_core_temps_c",
        "start_core_powers_w",
        "end_core_powers_w",
        "start_core_utilizations",
        "end_core_utilizations",
        "start_core_cpis",
        "end_core_cpis",
        "start_core_rel_nuca_cpis",
        "end_core_rel_nuca_cpis",
        "start_core_ips",
        "end_core_ips",
    ]

    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Detected columns were: {list(raw_df.columns)}"
        )

    X_rows = []
    Y_rows = []
    start_temp_rows = []
    end_temp_rows = []
    core_id_rows = []
    phase_rows = []
    group_rows = []
    kept_source_rows = []

    feature_names = None

    skipped_invalid_temp = 0
    skipped_parse_error = 0
    skipped_transition_phase = 0
    skipped_thread_lifecycle_change = 0
    repaired_negative_ips_values = 0

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

            start_ips_raw = parse_vec(row["start_core_ips"], colname="start_core_ips")
            repaired_negative_ips_values += int(np.sum(start_ips_raw < 0))

            start_ips = repair_ips_from_freq_cpi(
                raw_ips=start_ips_raw,
                freq_ghz=old_f,
                cpi=start_cpi,
            )
            log_start_ips = safe_log_ips(start_ips)

            start_thread_ids = None
            end_thread_ids = None
            if "start_core_thread_ids" in raw_df.columns and "end_core_thread_ids" in raw_df.columns:
                start_thread_ids = parse_vec(row["start_core_thread_ids"], colname="start_core_thread_ids")
                end_thread_ids = parse_vec(row["end_core_thread_ids"], colname="end_core_thread_ids")
                if active_thread_id_set(start_thread_ids) != active_thread_id_set(end_thread_ids):
                    skipped_thread_lifecycle_change += 1
                    continue

            (
                expected_active,
                expected_power,
                expected_util,
                expected_cpi,
                expected_ips,
            ) = infer_expected_runtime_from_thread_ids(
                old_f=old_f,
                new_f=new_f,
                active=active,
                start_power=start_power,
                start_util=start_util,
                start_cpi=start_cpi,
                start_ips=start_ips,
                start_thread_ids=start_thread_ids,
                end_thread_ids=end_thread_ids,
            )
            expected_ipc = cpi_to_ipc(expected_cpi)
            log_expected_ips = safe_log_ips(expected_ips)

            start_peak = float(row["start_peak_temp_c"])

            phase = classify_phase(start_util)
            if phase is None:
                skipped_transition_phase += 1
                continue

            for target_core in range(NUM_CORES):
                X, names = build_single_core_features(
                    target_core=target_core,
                    start_peak=start_peak,
                    old_f=old_f,
                    new_f=new_f,
                    delta_f=delta_f,
                    active=active,
                    expected_active=expected_active,
                    start_temp=start_temp,
                    start_power=start_power,
                    expected_power=expected_power,
                    start_util=start_util,
                    expected_util=expected_util,
                    start_ipc=start_ipc,
                    expected_ipc=expected_ipc,
                    expected_cpi=expected_cpi,
                    start_rel_nuca_cpi=start_rel_nuca_cpi,
                    log_start_ips=log_start_ips,
                    log_expected_ips=log_expected_ips,
                    phase=phase,
                )

                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise RuntimeError("Feature names changed across rows. This should not happen.")

                X_rows.append(X)
                Y_rows.append(float(delta_temp[target_core]))
                start_temp_rows.append(float(start_temp[target_core]))
                end_temp_rows.append(float(end_temp[target_core]))
                core_id_rows.append(target_core)
                phase_rows.append(phase)
                group_rows.append(idx)
                kept_source_rows.append(idx)

        except Exception:
            skipped_parse_error += 1
            continue

    if not X_rows:
        raise ValueError("No valid samples left after cleaning. Check input CSV.")

    cleaned_df = raw_df.loc[sorted(set(kept_source_rows))].copy()
    cleaned_df.insert(0, "source_row", sorted(set(kept_source_rows)))
    cleaned_df = cleaned_df.reset_index(drop=True)

    categorical_feature_indices = [
        feature_names.index("core_id"),
        feature_names.index("phase_id"),
    ]

    print("\nData cleaning")
    print("-------------")
    print(f"raw interval rows          : {len(raw_df)}")
    print(f"kept interval rows         : {len(cleaned_df)}")
    print(f"expanded core samples      : {len(X_rows)}")
    print(f"removed temp -1 rows       : {skipped_invalid_temp}")
    print(f"removed transition rows    : {skipped_transition_phase}")
    print(f"removed thread start/exit  : {skipped_thread_lifecycle_change}")
    print(f"removed parse errors       : {skipped_parse_error}")
    print(f"repaired negative IPS vals : {repaired_negative_ips_values}")

    phase_counts = pd.Series(phase_rows).value_counts()
    print("\nPhase counts")
    print("------------")
    for phase_name in PHASE_NAMES:
        print(f"{phase_name:15s}: {int(phase_counts.get(phase_name, 0))}")

    core_counts = pd.Series(core_id_rows).value_counts().sort_index()
    print("\nCore sample counts")
    print("------------------")
    for core_id in range(NUM_CORES):
        print(f"core {core_id}: {int(core_counts.get(core_id, 0))}")

    return DatasetBundle(
        X=np.stack(X_rows).astype(np.float32),
        Y=np.array(Y_rows, dtype=np.float32),
        start_temps=np.array(start_temp_rows, dtype=np.float32),
        end_temps=np.array(end_temp_rows, dtype=np.float32),
        core_ids=np.array(core_id_rows, dtype=np.int32),
        phases=np.array(phase_rows),
        groups=np.array(group_rows, dtype=np.int64),
        feature_names=feature_names,
        target_name="delta_temp",
        categorical_feature_indices=categorical_feature_indices,
        df=cleaned_df,
    )


# ------------------------------------------------------------
# Sample weighting
# ------------------------------------------------------------

def make_sample_weights(start_temps, Y_delta, args):
    end_temps = start_temps + Y_delta

    weights = np.ones(len(end_temps), dtype=np.float64)
    weights[end_temps > args.weight_t1] = args.weight_w1
    weights[end_temps > args.weight_t2] = args.weight_w2
    weights[end_temps > args.weight_t3] = args.weight_w3
    return weights


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(Y_true_delta, Y_pred_delta, start_temps):
    true_end_temp = start_temps + Y_true_delta
    pred_end_temp = start_temps + Y_pred_delta

    errors = pred_end_temp - true_end_temp
    abs_errors = np.abs(errors)

    return {
        "temp_rmse_c": rmse(true_end_temp, pred_end_temp),
        "temp_mae_c": mean_absolute_error(true_end_temp, pred_end_temp),
        "temp_r2": r2_score(true_end_temp, pred_end_temp),
        "temp_within_3c": np.mean(abs_errors <= 3.0),
        "temp_within_5c": np.mean(abs_errors <= 5.0),
        "temp_within_10c": np.mean(abs_errors <= 10.0),
        "bias_c": np.mean(errors),
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
            f"within_5c={m['temp_within_5c']:6.3f}  "
            f"bias={m['bias_c']:8.4f}"
        )


def print_metrics_by_core(title, Y_true, Y_pred, start_temps, core_ids):
    print("\n" + title)
    print("-" * len(title))

    for core_id in range(NUM_CORES):
        idx = np.where(core_ids == core_id)[0]
        if len(idx) == 0:
            continue

        m = compute_metrics(Y_true[idx], Y_pred[idx], start_temps[idx])
        print(
            f"core {core_id:<2d} "
            f"n={len(idx):5d}  "
            f"temp_mae={m['temp_mae_c']:8.4f}  "
            f"temp_rmse={m['temp_rmse_c']:8.4f}  "
            f"within_5c={m['temp_within_5c']:6.3f}  "
            f"bias={m['bias_c']:8.4f}"
        )


def print_metrics_by_temp_band(title, Y_true, Y_pred, start_temps):
    true_end_temp = start_temps + Y_true
    pred_end_temp = start_temps + Y_pred

    bands = [
        ("< 60 C   ", true_end_temp < 60),
        ("60-80 C  ", (true_end_temp >= 60) & (true_end_temp < 80)),
        ("80-95 C  ", (true_end_temp >= 80) & (true_end_temp < 95)),
        ("95-110 C ", (true_end_temp >= 95) & (true_end_temp < 110)),
        (">= 110 C ", true_end_temp >= 110),
    ]

    print("\n" + title)
    print("-" * len(title))
    print(f"{'band':12s} {'n':>6s}  {'mae':>8s}  {'rmse':>8s}  {'bias':>8s}")

    for label, mask in bands:
        n = int(np.sum(mask))
        if n == 0:
            print(f"{label:12s} {n:>6d}  {'-':>8s}  {'-':>8s}  {'-':>8s}")
            continue

        err = pred_end_temp[mask] - true_end_temp[mask]
        mae = np.mean(np.abs(err))
        rms = math.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        print(f"{label:12s} {n:>6d}  {mae:>8.3f}  {rms:>8.3f}  {bias:>8.3f}")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

def make_model(args):
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


def fit_model(X_train, Y_train, sample_weight, categorical_indices, args):
    model = make_model(args)
    model.fit(
        X_train,
        Y_train,
        sample_weight=sample_weight,
        categorical_feature=categorical_indices,
    )
    return model


def predict_model(model, X):
    return model.booster_.predict(X).astype(np.float32)


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------

def train_and_predict_fold(bundle, train_idx, test_idx, args):
    X_train = bundle.X[train_idx]
    X_test = bundle.X[test_idx]

    Y_train = bundle.Y[train_idx]
    Y_test = bundle.Y[test_idx]

    start_train = bundle.start_temps[train_idx]
    start_test = bundle.start_temps[test_idx]

    sw = make_sample_weights(start_train, Y_train, args)

    model = fit_model(
        X_train,
        Y_train,
        sw,
        bundle.categorical_feature_indices,
        args,
    )

    Y_train_pred = predict_model(model, X_train)
    Y_test_pred = predict_model(model, X_test)

    train_metrics = compute_metrics(Y_train, Y_train_pred, start_train)
    test_metrics = compute_metrics(Y_test, Y_test_pred, start_test)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model": model,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_train_pred": Y_train_pred,
        "Y_test_pred": Y_test_pred,
        "start_train": start_train,
        "start_test": start_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def run_cross_validation(bundle, args):
    n_samples = len(bundle.X)
    unique_groups = np.unique(bundle.groups)

    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 original interval rows for GroupKFold.")

    folds = min(args.folds, len(unique_groups))
    gkf = GroupKFold(n_splits=folds)

    train_metrics_all = []
    test_metrics_all = []

    all_test_y = []
    all_test_pred = []
    all_test_start = []
    all_test_phase = []
    all_test_core_id = []

    for fold_id, (train_idx, test_idx) in enumerate(
        gkf.split(bundle.X, bundle.Y, groups=bundle.groups),
        start=1,
    ):
        print(f"\nFold {fold_id}/{folds}")
        print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")
        print(f"train groups : {len(np.unique(bundle.groups[train_idx]))}, test groups: {len(np.unique(bundle.groups[test_idx]))}")

        result = train_and_predict_fold(bundle, train_idx, test_idx, args)

        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]

        train_metrics_all.append(train_metrics)
        test_metrics_all.append(test_metrics)

        all_test_y.append(result["Y_test"])
        all_test_pred.append(result["Y_test_pred"])
        all_test_start.append(result["start_test"])
        all_test_phase.append(bundle.phases[test_idx])
        all_test_core_id.append(bundle.core_ids[test_idx])

        print(f"test temp_rmse_c : {test_metrics['temp_rmse_c']:.4f}")
        print(f"test temp_mae_c  : {test_metrics['temp_mae_c']:.4f}")
        print(f"test temp_r2     : {test_metrics['temp_r2']:.4f}")
        print(f"test within_5c   : {test_metrics['temp_within_5c']:.4f}")
        print(f"test bias_c      : {test_metrics['bias_c']:.4f}")

    print_metric_table("TRAIN metrics across folds", train_metrics_all)
    print_metric_table("TEST / CV metrics across folds", test_metrics_all)

    all_test_y = np.concatenate(all_test_y, axis=0)
    all_test_pred = np.concatenate(all_test_pred, axis=0)
    all_test_start = np.concatenate(all_test_start, axis=0)
    all_test_phase = np.concatenate(all_test_phase, axis=0)
    all_test_core_id = np.concatenate(all_test_core_id, axis=0)

    print_metrics_by_phase(
        "Combined TEST metrics by phase",
        all_test_y,
        all_test_pred,
        all_test_start,
        all_test_phase,
    )

    print_metrics_by_core(
        "Combined TEST metrics by target core",
        all_test_y,
        all_test_pred,
        all_test_start,
        all_test_core_id,
    )

    print_metrics_by_temp_band(
        "Combined TEST metrics by true-end-temp band",
        all_test_y,
        all_test_pred,
        all_test_start,
    )

    return train_metrics_all, test_metrics_all


def train_final_model(bundle, args):
    sw = make_sample_weights(bundle.start_temps, bundle.Y, args)

    model = fit_model(
        bundle.X,
        bundle.Y,
        sw,
        bundle.categorical_feature_indices,
        args,
    )

    Y_pred = predict_model(model, bundle.X)

    train_all_metrics = compute_metrics(bundle.Y, Y_pred, bundle.start_temps)

    saved = {
        "model": model,
        "feature_names": bundle.feature_names,
        "target_name": bundle.target_name,
        "phase_names": PHASE_NAMES,
        "phase_to_int": PHASE_TO_INT,
        "categorical_feature_indices": bundle.categorical_feature_indices,
        "model_type": "lightgbm_single_core_delta",
        "prediction_kind": "end_temp = start_temp + predicted_delta",
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(saved, args.output)

    cpp_output = args.cpp_output
    if cpp_output is None:
        output_root, _ = os.path.splitext(args.output)
        cpp_output = output_root + ".txt"

    export_cpp_model(saved, cpp_output)

    print("\nFINAL model trained on all cleaned core samples")
    print("-----------------------------------------------")
    for k, v in train_all_metrics.items():
        print(f"{k:28s} {v:10.4f}")

    print_metrics_by_phase(
        "FINAL train-all metrics by phase",
        bundle.Y,
        Y_pred,
        bundle.start_temps,
        bundle.phases,
    )

    print_metrics_by_core(
        "FINAL train-all metrics by target core",
        bundle.Y,
        Y_pred,
        bundle.start_temps,
        bundle.core_ids,
    )

    print_metrics_by_temp_band(
        "FINAL train-all metrics by true-end-temp band",
        bundle.Y,
        Y_pred,
        bundle.start_temps,
    )

    print_feature_importance(model, bundle.feature_names, top_k=25)

    print(f"\nSaved Python model to: {args.output}")
    print(f"Saved C++ text model to: {cpp_output}")
    print(f"Saved C++ JSON model to: {os.path.splitext(cpp_output)[0] + '.json'}")

    return saved, train_all_metrics


def print_feature_importance(model, feature_names, top_k=25):
    booster = model.booster_
    importances = booster.feature_importance(importance_type="gain")

    order = np.argsort(-importances)

    print(f"\nTop {top_k} features by gain")
    print("-" * 48)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:3d}. {feature_names[idx]:35s}  gain={importances[idx]:12.1f}")


# ------------------------------------------------------------
# C++ export
# ------------------------------------------------------------

def dump_tree_to_dict(tree_dict):
    nodes = []

    def add_node(node):
        node_idx = len(nodes)
        nodes.append(None)

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
            threshold_raw = node["threshold"]
            if isinstance(threshold_raw, str):
                cat_values = [int(x) for x in threshold_raw.split("||")]
            else:
                cat_values = [int(threshold_raw)]

            entry = {
                "is_leaf": False,
                "leaf_value": 0.0,
                "split_feature": split_feature,
                "threshold": 0.0,
                "left": -1,
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
    model = saved["model"]
    feature_names = saved["feature_names"]
    cat_indices = saved["categorical_feature_indices"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    booster = model.booster_
    dump = booster.dump_model()
    trees = dump["tree_info"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("THERMAL_LGBM_SINGLE_CORE_MODEL_V1\n")
        f.write(f"num_features {len(feature_names)}\n")
        f.write("target_name delta_temp\n")
        f.write("prediction_kind end_temp_equals_start_temp_plus_predicted_delta\n")

        f.write("phase_names\n")
        for name in saved["phase_names"]:
            f.write(f"{name}\n")

        f.write("feature_names\n")
        for name in feature_names:
            f.write(f"{name}\n")

        f.write("categorical_feature_indices\n")
        f.write(" ".join(str(i) for i in cat_indices) + "\n")

        f.write(f"num_trees {len(trees)}\n")

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

    json_path = os.path.splitext(output_path)[0] + ".json"
    json_dump = {
        "model_format": "THERMAL_LGBM_SINGLE_CORE_MODEL_V1",
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "target_name": "delta_temp",
        "prediction_kind": "end_temp = start_temp + predicted_delta",
        "phase_names": saved["phase_names"],
        "phase_to_int": saved["phase_to_int"],
        "categorical_feature_indices": cat_indices,
        "trees": [dump_tree_to_dict(t) for t in trees],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dump, f)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, help="Path to input CSV/TSV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        default="ml_models/lgbm_single_core_thermal_model.joblib",
    )
    parser.add_argument(
        "--cpp-output",
        default=None,
        help="Where to save the C++-readable text model. Default: output path with .txt suffix",
    )

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-samples", type=int, default=10)

    parser.add_argument("--reg-lambda", type=float, default=0.1)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)

    parser.add_argument("--weight-t1", type=float, default=70.0)
    parser.add_argument("--weight-w1", type=float, default=2.0)
    parser.add_argument("--weight-t2", type=float, default=85.0)
    parser.add_argument("--weight-w2", type=float, default=5.0)
    parser.add_argument("--weight-t3", type=float, default=100.0)
    parser.add_argument("--weight-w3", type=float, default=10.0)

    parser.add_argument("--print-features", action="store_true")
    parser.add_argument("--no-cv", action="store_true")

    args = parser.parse_args()

    bundle = build_dataset(args.csv)

    print("\nLoaded dataset")
    print("--------------")
    print(f"core samples   : {bundle.X.shape[0]}")
    print(f"input features : {bundle.X.shape[1]}")
    print(f"target         : {bundle.target_name}")
    print(f"categorical idx: {bundle.categorical_feature_indices}")
    print(f"categoricals   : {[bundle.feature_names[i] for i in bundle.categorical_feature_indices]}")

    print("\nLightGBM hyperparameters")
    print("------------------------")
    print(f"n_estimators       : {args.n_estimators}")
    print(f"learning_rate      : {args.learning_rate}")
    print(f"num_leaves         : {args.num_leaves}")
    print(f"max_depth          : {args.max_depth}")
    print(f"min_child_samples  : {args.min_child_samples}")
    print(f"reg_lambda         : {args.reg_lambda}")
    print(f"subsample          : {args.subsample}")
    print(f"colsample_bytree   : {args.colsample_bytree}")

    print("\nSample weighting")
    print("----------------")
    print(f"end temp > {args.weight_t1:5.1f} C  -> weight {args.weight_w1}")
    print(f"end temp > {args.weight_t2:5.1f} C  -> weight {args.weight_w2}")
    print(f"end temp > {args.weight_t3:5.1f} C  -> weight {args.weight_w3}")

    if args.print_features:
        print("\nFeature names")
        print("-------------")
        for i, name in enumerate(bundle.feature_names):
            marker = "  categorical" if i in bundle.categorical_feature_indices else ""
            print(f"{i:03d}: {name}{marker}")
        return

    if not args.no_cv:
        run_cross_validation(bundle, args)

    train_final_model(bundle, args)


if __name__ == "__main__":
    main()
