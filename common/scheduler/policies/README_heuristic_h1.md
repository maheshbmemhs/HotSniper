# Heuristic H1 Migration Policy

Enable the policy with:

```ini
[scheduler/open/migration]
logic = heuristicH1
epoch = 1000000

[scheduler/open/heuristic_h1]
num_states = 4
state_value = 1.0,2.0,3.0,4.0
core_state = 0,1,2,3
target_ips = 10.0
profile_file = /absolute/path/to/profiles.tsv
debug = false
```

`state_value` is the enabled fixed frequency state list in GHz. `core_state` maps each physical core to one state index, so the example fixes core 0 at 1.0 GHz, core 1 at 2.0 GHz, core 2 at 3.0 GHz, and core 3 at 4.0 GHz. `target_ips` uses the same unit as the profile, billions of instructions per second.

The profile file is tab-separated or whitespace-separated:

```text
name state ips cpi temp power
blackscholes-simsmall-2 1.00 1.46 0.66 53.03 0.85
blackscholes-simsmall-2 2.00 2.94 0.66 59.53 1.35
blackscholes-simsmall-2 3.00 4.33 0.69 65.30 2.10
blackscholes-simsmall-2 4.00 5.71 0.70 71.00 3.05
```

Only benchmarks with rows for every enabled state are used for nearest-neighbor prediction. If none match, H1 falls back to measured IPS on the current state, scales IPS by frequency ratio for other states, and uses measured current power or 0 W.

Optional fixed per-core frequencies:

```ini
[scheduler/open/dvfs]
logic = fixedStates
dvfs_epoch = 1000000

[scheduler/open/dvfs/fixed_states]
frequency = 1000,2000,3000,4000
```
