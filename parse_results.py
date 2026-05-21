#!/usr/bin/env python3
"""
Parse HotSniper result folders into a unified metrics table.

Usage:
    # Parse already-extracted result folders:
    python3 parse_results.py /path/to/results/

    # Extract all zips in a folder first, then parse:
    python3 parse_results.py /path/to/zips/ --unzip

    # Output to CSV:
    python3 parse_results.py /path/to/zips/ --unzip --csv output.csv
"""

import os, re, gzip, sys, csv, zipfile
from collections import OrderedDict


def find_result_dirs(root):
    dirs = []
    for dirpath, _, files in os.walk(root):
        if 'sim.out' in files:
            dirs.append(dirpath)
    return sorted(dirs)


def parse_simout(path):
    with open(path) as f:
        text = f.read()
    def row(name):
        m = re.search(r'^\s*' + re.escape(name) + r'\s*\|([^\n]+)', text, re.M)
        if not m:
            return None
        return [p.strip() for p in m.group(1).split('|') if p.strip()]
    return {
        'instructions': [int(x) for x in row('Instructions')],
        'cycles':        [int(x) for x in row('Cycles')],
        'time_ns':       [int(x) for x in row('Time (ns)')],
        'ipc':           [float(x) for x in row('IPC')],
    }


def parse_executioninfo(path):
    info = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def parse_response_time(execlog_gz):
    with gzip.open(execlog_gz, 'rt', errors='ignore') as f:
        text = f.read()
    m = re.search(r'Average Response Time \(ns\)\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    matches = re.findall(r'Task \d+ \(Response/Service/Wait\) Time \(ns\)\s*:\s*(\d+)', text)
    if matches:
        vals = [int(x) for x in matches]
        return sum(vals) / len(vals)
    return None


def parse_periodic_log(gz_path):
    rows, header = [], None
    with gzip.open(gz_path, 'rt', errors='ignore') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if i == 0:
                header = parts
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    return header, rows


def compute_power(header, rows):
    """Average chip power = mean over time of (L3 + sum of all core sub-units)."""
    core_idx = {c: [] for c in range(4)}
    l3_idx = None
    for i, h in enumerate(header):
        if h == 'L3':
            l3_idx = i
        else:
            m = re.match(r'C_(\d)_', h)
            if m:
                core_idx[int(m.group(1))].append(i)
    series = []
    for r in rows:
        p = (r[l3_idx] if l3_idx is not None else 0.0)
        p += sum(r[i] for c in range(4) for i in core_idx[c])
        series.append(p)
    return sum(series) / len(series) if series else None


def compute_temp(header, rows):
    """
    Peak temp  = max over all core sub-units over all time steps.
    Avg chip temp = per-core mean of sub-units, then averaged over 4 cores
                   (L3 excluded — it sits ~45°C and dilutes the metric).
    """
    core_idx = {c: [] for c in range(4)}
    for i, h in enumerate(header):
        m = re.match(r'C_(\d)_', h)
        if m:
            core_idx[int(m.group(1))].append(i)

    peak = 0.0
    avg_sum = {c: 0.0 for c in range(4)}
    n = len(rows)
    for r in rows:
        for c in range(4):
            for i in core_idx[c]:
                if r[i] > peak:
                    peak = r[i]
            avg_sum[c] += sum(r[i] for i in core_idx[c]) / len(core_idx[c])

    avg_chip = sum(avg_sum[c] / n for c in range(4)) / 4 if n else None
    return peak if peak > 0 else None, avg_chip


def label(dirname):
    name = os.path.basename(dirname).lower()
    if 'ondemand' in name:   return 'onDemand'
    if 'coldestcore' in name: return 'ColdestCore'
    if 'fixedvf' in name:    return 'FixedVF-SOTA'
    if 'percore' in name:    return 'PerCoreDVFS-SOTA'
    if 'fixedstates' in name: return 'FixedStates'
    return '?'


def process_run(rundir):
    sim   = parse_simout(os.path.join(rundir, 'sim.out'))
    info  = parse_executioninfo(os.path.join(rundir, 'executioninfo.txt')) \
            if os.path.exists(os.path.join(rundir, 'executioninfo.txt')) else {}

    per_core_ips = [instr / (t / 1e9) if t > 0 else 0.0
                    for instr, t in zip(sim['instructions'], sim['time_ns'])]
    agg_ips = sum(per_core_ips)

    execlog = os.path.join(rundir, 'execution.log.gz')
    avg_rt  = parse_response_time(execlog) if os.path.exists(execlog) else None

    power_path = os.path.join(rundir, 'PeriodicPower.log.gz')
    therm_path = os.path.join(rundir, 'PeriodicThermal.log.gz')

    avg_power = peak_temp = avg_temp = energy = None

    if os.path.exists(power_path):
        ph, prows = parse_periodic_log(power_path)
        avg_power = compute_power(ph, prows)

    if os.path.exists(therm_path):
        th, trows = parse_periodic_log(therm_path)
        peak_temp, avg_temp = compute_temp(th, trows)

    if avg_power is not None and sim['time_ns']:
        energy = avg_power * max(sim['time_ns']) / 1e9

    return {
        'governor':    label(rundir),
        'tasks':       info.get('tasks', '?'),
        'rt_ms':       avg_rt / 1e6 if avg_rt else None,
        'peak_C':      peak_temp,
        'avg_C':       avg_temp,
        'power_W':     avg_power,
        'energy_J':    energy,
        'ips_total':   agg_ips / 1e6,
        'ips_cores':   [x / 1e6 for x in per_core_ips],
    }


def print_table(results):
    hdrs = ['#', 'Governor', 'Tasks',
            'Resp(ms)', 'Peak(°C)', 'AvgTemp(°C)',
            'Power(W)', 'Energy(J)', 'IPS_total',
            'IPS_C0', 'IPS_C1', 'IPS_C2', 'IPS_C3']
    rows = []
    for i, r in enumerate(results, 1):
        ips = r['ips_cores']
        rows.append([
            i, r['governor'], r['tasks'],
            f"{r['rt_ms']:.2f}"    if r['rt_ms']    is not None else '?',
            f"{r['peak_C']:.2f}"   if r['peak_C']   is not None else '?',
            f"{r['avg_C']:.2f}"    if r['avg_C']    is not None else '?',
            f"{r['power_W']:.2f}"  if r['power_W']  is not None else '?',
            f"{r['energy_J']:.3f}" if r['energy_J'] is not None else '?',
            f"{r['ips_total']:.0f}",
            *[f"{x:.0f}" for x in ips],
        ])
    colw = [max(len(str(row[i])) for row in [hdrs]+rows) for i in range(len(hdrs))]
    fmt  = ' | '.join('{:<'+str(w)+'}' for w in colw)
    print(fmt.format(*hdrs))
    print('-+-'.join('-'*w for w in colw))
    for row in rows:
        print(fmt.format(*row))


def write_csv(results, path):
    fields = ['governor','tasks','avg_response_time_ms','peak_temp_C',
              'avg_chip_temp_C','avg_power_W','energy_J','total_ips_MIPS',
              'ips_core0_MIPS','ips_core1_MIPS','ips_core2_MIPS','ips_core3_MIPS']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            ips = r['ips_cores']
            w.writerow({
                'governor':             r['governor'],
                'tasks':                r['tasks'],
                'avg_response_time_ms': f"{r['rt_ms']:.3f}"    if r['rt_ms']    is not None else '',
                'peak_temp_C':          f"{r['peak_C']:.2f}"   if r['peak_C']   is not None else '',
                'avg_chip_temp_C':      f"{r['avg_C']:.2f}"    if r['avg_C']    is not None else '',
                'avg_power_W':          f"{r['power_W']:.3f}"  if r['power_W']  is not None else '',
                'energy_J':             f"{r['energy_J']:.4f}" if r['energy_J'] is not None else '',
                'total_ips_MIPS':       f"{r['ips_total']:.1f}",
                'ips_core0_MIPS':       f"{ips[0]:.1f}",
                'ips_core1_MIPS':       f"{ips[1]:.1f}",
                'ips_core2_MIPS':       f"{ips[2]:.1f}",
                'ips_core3_MIPS':       f"{ips[3]:.1f}",
            })
    print(f"\nCSV saved → {path}")


def unzip_all(zip_dir, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    zips = sorted(f for f in os.listdir(zip_dir) if f.endswith('.zip'))
    if not zips:
        print(f"No .zip files found in {zip_dir}")
        return
    for z in zips:
        dest = os.path.join(extract_to, z[:-4])
        print(f"  Extracting {z} ...")
        with zipfile.ZipFile(os.path.join(zip_dir, z)) as zf:
            zf.extractall(dest)
    print(f"Extracted {len(zips)} zip(s).\n")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Parse HotSniper results.')
    p.add_argument('root', nargs='?', default='.',
                   help='Folder of extracted result dirs, or folder of .zip files (with --unzip)')
    p.add_argument('--unzip',  action='store_true',
                   help='Unzip all .zip files in root first, then parse')
    p.add_argument('--csv', metavar='FILE.csv',
                   help='Write results to CSV')
    args = p.parse_args()

    scan_root = args.root
    if args.unzip:
        extract_to = os.path.join(args.root, '_extracted')
        unzip_all(args.root, extract_to)
        scan_root = extract_to

    rundirs = find_result_dirs(scan_root)
    print(f"Found {len(rundirs)} result folder(s).\n")

    results = []
    for d in rundirs:
        try:
            results.append(process_run(d))
        except Exception as e:
            print(f"  [skipped {os.path.basename(d)}: {e}]")

    print_table(results)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == '__main__':
    main()