# H1 Fixed-VF Dynamic Mapping Experiment

This experiment compares three 4-core modes:

1. `4.0GHz + maxFreq`: all cores at 4 GHz, upper-bound performance baseline.
2. `fixedStates`: core 0/1/2/3 fixed at 1/2/3/4 GHz, no migration.
3. `heuristicH1 + fixedStates`: same fixed core frequencies, but H1 periodically migrates threads between cores.

Optionally add `--include-pcgov` to also run the existing dynamic DVFS governor.

Run from the repository root:

```bash
python3 simulationcontrol/run.py h1 \
  --benchmark parsec-blackscholes \
  --parallelism 4 \
  --input-set simsmall \
  --profile-file /absolute/path/to/profiles.tsv \
  --target-ips 10.0
```

For a smoke test, omit `--profile-file`; the script will use `common/scheduler/policies/heuristic_h1_profiles.example.tsv`. Use a measured profile for real results.

The H1 profile format is whitespace- or tab-separated:

```text
name state ips cpi temp power
blackscholes-simsmall-2 1.00 1.46 0.66 53.03 0.85
blackscholes-simsmall-2 2.00 2.94 0.66 59.53 1.35
blackscholes-simsmall-2 3.00 4.33 0.69 65.30 2.10
blackscholes-simsmall-2 4.00 5.71 0.70 71.00 3.05
```

`ips` and `target_ips` are in billions of instructions per second.

Useful outputs land in `results/results_<timestamp>_*`:

- `execution.log.gz`: H1 migration logs, including `[MigrationH1]` lines.
- `PeriodicFrequency.log.gz`: verify fixed 1/2/3/4 GHz states.
- `PeriodicPower.log.gz`: per-component power.
- `sim.out` and `sim.stats.sqlite3`: performance counters and summary stats.

For more verbose H1 prediction logs:

```bash
python3 simulationcontrol/run.py h1 --debug-h1 --profile-file /absolute/path/to/profiles.tsv
```
