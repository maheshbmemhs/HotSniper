import re
import gzip
import os
import pandas as pd
import io
import json
from collections import OrderedDict

core_names = ["C_0", "C_1", "C_2", "C_3"]
num_cores = 4


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _touch_file(path):
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8"):
        pass


def _open_file(run, filename):
    os.makedirs(run, exist_ok=True)

    full_filename = os.path.join(run, filename)
    if os.path.exists(full_filename):
        return open(full_filename, "r", encoding="utf-8")

    gzip_filename = "{}.gz".format(full_filename)
    if os.path.exists(gzip_filename):
        return io.TextIOWrapper(gzip.open(gzip_filename, "r"), encoding="utf-8")

    # 缺文件就创建空文件
    _touch_file(full_filename)
    return open(full_filename, "r", encoding="utf-8")


def get_file(run, filename):
    os.makedirs(run, exist_ok=True)

    full_filename = os.path.join(run, filename)
    if os.path.exists(full_filename):
        return full_filename

    gzip_filename = "{}.gz".format(full_filename)
    if os.path.exists(gzip_filename):
        return gzip_filename

    # 缺文件就创建空文件
    _touch_file(full_filename)
    return full_filename


def load_json_file(path):
    # 不存在才创建，存在就绝对不清空
    if not os.path.exists(path):
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"experiments": []}, f, indent=4)

    # 空文件写一个默认结构
    if os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"experiments": []}, f, indent=4)

    # JSON 必须用 r 读，不能用 w
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError:
            print("Warning: {} is not valid JSON. Use empty experiments.".format(path))
            return {"experiments": []}


def _safe_read_csv_whitespace(filename):
    try:
        if os.path.getsize(filename) == 0:
            return pd.DataFrame()

        return pd.read_csv(filename, delim_whitespace=True)
    except Exception:
        return pd.DataFrame()


def _mean_or_zero(series):
    try:
        val = series.mean()
        if pd.isna(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def get_average_response_time(run):
    with _open_file(run, "execution.log") as f:
        for line in f:
            m = re.search(r"Average Response Time \(ns\)\s+:\s+(\d+)", line)
            if m is not None:
                return int(m.group(1))
    return 0


def _columns_for_core(df, core):
    cols = []

    for col in df.columns:
        s = str(col)

        if s == core:
            cols.append(col)
        elif s.startswith(core + "_"):
            cols.append(col)
        elif s.startswith(core + "-"):
            cols.append(col)

    return cols


def aggregate_core_metric(df, atype="sum"):
    """
    把 PeriodicThermal.log / PeriodicPower.log 这种日志聚合成：
    {
        "C_0": value,
        "C_1": value,
        "C_2": value,
        "C_3": value,
        "L3": value
    }

    atype:
    - max: 每个时间点，对 core 的 subcomponents 取 max，再对时间求平均
    - min: 每个时间点，对 core 的 subcomponents 取 min，再对时间求平均
    - sum: 每个时间点，对 core 的 subcomponents 求和，再对时间求平均
    """

    result = OrderedDict()

    if df is None or df.empty:
        for core in core_names:
            result[core] = 0.0
        result["L3"] = 0.0
        return result

    for core in core_names:
        cols = _columns_for_core(df, core)

        if len(cols) == 0:
            result[core] = 0.0
            continue

        if atype == "max":
            per_row = df[cols].max(axis=1)
        elif atype == "min":
            per_row = df[cols].min(axis=1)
        else:
            per_row = df[cols].sum(axis=1)

        result[core] = _mean_or_zero(per_row)

    if "L3" in df.columns:
        result["L3"] = _mean_or_zero(df["L3"])
    else:
        result["L3"] = 0.0

    return result


def get_temperature_metrics(run_path):
    filename = get_file(run_path, "PeriodicThermal.log")
    df = _safe_read_csv_whitespace(filename)

    vals = aggregate_core_metric(df, atype="max")

    core_vals = []
    for core in core_names:
        core_vals.append(vals.get(core, 0.0))

    l3 = vals.get("L3", 0.0)

    # 原逻辑：4 个 core + L3 一起平均
    total_avg = (sum(core_vals) + l3) / float(len(core_vals) + 1)

    return core_vals, total_avg


def get_power_metrics(run_path):
    filename = get_file(run_path, "PeriodicPower.log")
    df = _safe_read_csv_whitespace(filename)

    vals = aggregate_core_metric(df, atype="sum")

    core_vals = []
    for core in core_names:
        core_vals.append(vals.get(core, 0.0))

    l3 = vals.get("L3", 0.0)

    # 原逻辑：4 个 core 功耗 + L3
    total_sum = sum(core_vals) + l3

    return core_vals, total_sum


def get_energy_metrics(run_path):
    filename = get_file(run_path, "PeriodicPower.log")
    df = _safe_read_csv_whitespace(filename)

    vals = aggregate_core_metric(df, atype="sum")

    rsp_time = get_average_response_time(run_path)
    time_s = rsp_time / 1e9

    core_energy = []
    for core in core_names:
        power_avg = vals.get(core, 0.0)
        core_energy.append(power_avg * time_s)

    l3_power = vals.get("L3", 0.0)
    l3_energy = l3_power * time_s

    total_energy = sum(core_energy) + l3_energy

    return core_energy, total_energy


def get_cpi_stack_part_trace(run, part="total"):
    trace_values = []

    with _open_file(run, "PeriodicCPIStack.log") as f:
        f.readline()

        for line in f:
            if line.startswith(part + "\t"):
                items = line.split()[1:]

                if items == ["-"]:
                    ps = [0.0] * num_cores
                else:
                    ps = []

                    for value in items:
                        try:
                            ps.append(float(value))
                        except Exception:
                            ps.append(0.0)

                    while len(ps) < num_cores:
                        ps.append(0.0)

                    ps = ps[:num_cores]

                trace_values.append(ps)

    if len(trace_values) == 0:
        return [[] for _ in range(num_cores)]

    return list(zip(*trace_values))


def get_cpi_traces(run, raw=False):
    traces = list(map(list, get_cpi_stack_part_trace(run, "total")))

    if not raw:
        w = 2

        for trace in traces:
            drop = []

            for i in range(len(trace)):
                left = max(i - w, 0)
                right = min(i + w, len(trace))

                if any(t > 20 for t in trace[left:right]):
                    drop.append(i)

            for i in drop:
                trace[i] = None

    return traces


def get_cpi_metrics(run_path):
    # 保留这个调用：缺 cpi-stack.txt 时会自动创建空文件
    get_file(run_path, "cpi-stack.txt")

    traces = get_cpi_traces(run_path)

    core_cpi = []
    worker_avg = 0.0
    worker_count = 0

    for core in range(num_cores):
        if core < len(traces):
            trace = traces[core]
        else:
            trace = []

        valid_trace = [value for value in trace if value is not None]

        if len(valid_trace) > 0:
            avg = sum(valid_trace) / float(len(valid_trace))
        else:
            avg = 0.0

        core_cpi.append(avg)

        # 原逻辑：worker 是 core > 0
        if core > 0:
            worker_avg += avg
            worker_count += 1

    if worker_count > 0:
        worker_avg = worker_avg / float(worker_count)
    else:
        worker_avg = 0.0

    return core_cpi, worker_avg


def create_experiment_summary_csv(experiment):
    runs = experiment.get("runs", [])
    out_path = experiment.get("output_dir", "./txt_out/")
    name = experiment.get("name", "unknown")

    os.makedirs(out_path, exist_ok=True)

    rows = []

    for run in runs:
        run_path = run.get("path", "")
        run_name = run.get("name", "")

        row = OrderedDict()

        row["Run"] = run_name
        row["resp_time_ns"] = get_average_response_time(run_path)

        temp_vals, temp_total_avg = get_temperature_metrics(run_path)
        for i, core in enumerate(core_names):
            row["temp_" + core] = temp_vals[i]
        row["temp_total_avg"] = temp_total_avg

        power_vals, power_total_sum = get_power_metrics(run_path)
        for i, core in enumerate(core_names):
            row["power_" + core] = power_vals[i]
        row["power_total_sum"] = power_total_sum

        energy_vals, energy_total = get_energy_metrics(run_path)
        for i, core in enumerate(core_names):
            row["energy_" + core] = energy_vals[i]
        row["energy_total"] = energy_total

        cpi_vals, cpi_avg_workers = get_cpi_metrics(run_path)
        for i, core in enumerate(core_names):
            row["cpi_" + core] = cpi_vals[i]
        row["cpi_avg_workers"] = cpi_avg_workers

        rows.append(row)

    columns = ["Run", "resp_time_ns"]

    for core in core_names:
        columns.append("temp_" + core)
    columns.append("temp_total_avg")

    for core in core_names:
        columns.append("power_" + core)
    columns.append("power_total_sum")

    for core in core_names:
        columns.append("energy_" + core)
    columns.append("energy_total")

    for core in core_names:
        columns.append("cpi_" + core)
    columns.append("cpi_avg_workers")

    df = pd.DataFrame(rows, columns=columns)

    out_file = os.path.join(out_path, "summary-" + name + ".csv")
    df.to_csv(out_file, index=False, float_format="%.2f")

    print("Wrote {}".format(out_file))


def create_data(experiments):
    if experiments is None:
        experiments = {"experiments": []}

    if "experiments" not in experiments:
        experiments["experiments"] = []

    for experiment in experiments["experiments"]:
        create_experiment_summary_csv(experiment)


if __name__ == "__main__":
    blackscholes_experiments = load_json_file("blackscholes_expr.json")
    create_data(blackscholes_experiments)

    streamcluster_experiments = load_json_file("streamcluster_expr.json")
    create_data(streamcluster_experiments)