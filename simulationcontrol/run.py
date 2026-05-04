import argparse
import datetime
import math
import os
import gzip
import platform
import random
import re
import shutil
import subprocess
import time
import traceback
import sys


from config import NUMBER_CORES, RESULTS_FOLDER, SNIPER_CONFIG, SCRIPTS, ENABLE_HEARTBEATS

try:
    from resultlib.plot import create_plots
    PLOT_IMPORT_ERROR = None
except Exception as e:
    create_plots = None
    PLOT_IMPORT_ERROR = e

HERE = os.path.dirname(os.path.abspath(__file__))
SNIPER_BASE = os.path.dirname(HERE)
BENCHMARKS = os.path.join(SNIPER_BASE, 'benchmarks')
BATCH_START = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')


def change_base_configuration(base_configuration):
    base_cfg = os.path.join(SNIPER_BASE, 'config/base.cfg')
    with open(base_cfg, 'r') as f:
        content = f.read()
    with open(base_cfg, 'w') as f:
        for line in content.splitlines():
            m = re.match('.*cfg:(!?)([a-zA-Z_\\.0-9]+)$', line)
            if m:
                inverted = m.group(1) == '!'
                include = inverted ^ (m.group(2) in base_configuration)
                included = line[0] != '#'
                if include and not included:
                    line = line[1:]
                elif not include and included:
                    line = '#' + line
            f.write(line)
            f.write('\n')


def format_cfg_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def change_base_configuration_values(overrides):
    """Override active base.cfg values by full config path.

    Example key: scheduler/open/heuristic_h1/profile_file
    This keeps the cfg: toggling mechanism intact, then applies experiment-specific
    values such as target IPS and profile path.
    """
    if not overrides:
        return

    base_cfg = os.path.join(SNIPER_BASE, 'config/base.cfg')
    with open(base_cfg, 'r') as f:
        lines = f.read().splitlines()

    current_section = ''
    pending = dict((k, format_cfg_value(v)) for k, v in overrides.items())
    output = []

    for line in lines:
        stripped = line.strip()
        section_match = re.match(r'^\[([^\]]+)\]\s*$', stripped)
        if section_match:
            current_section = section_match.group(1)
            output.append(line)
            continue

        if not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            full_key = '{}/{}'.format(current_section, key) if current_section else key
            if full_key in pending:
                indent = line[:len(line) - len(line.lstrip())]
                line = '{}{} = {}'.format(indent, key, pending.pop(full_key))

        output.append(line)

    if pending:
        output.append('')
        output.append('# Experiment overrides added by simulationcontrol/run.py')
        for full_key, value in sorted(pending.items()):
            section, key = full_key.rsplit('/', 1)
            output.append('[{}]'.format(section))
            output.append('{} = {}'.format(key, value))

    with open(base_cfg, 'w') as f:
        f.write('\n'.join(output))
        f.write('\n')


def prev_run_cleanup():
    '''Cleanup files potentially left over from aborted previous runs.'''

    pattern = r"^\d+\.hb.log$" # Heartbeat logs
    for f in os.listdir(BENCHMARKS):
        if not re.match(pattern, f):
            continue

        file_path = os.path.join(BENCHMARKS, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for f in os.listdir(BENCHMARKS):
        if ('output.' in f) or ('.264' in f) or ('poses.' in f) or ('app_mapping' in f) :
            os.remove(os.path.join(BENCHMARKS, f))
        

def save_output(base_configuration, benchmark, console_output, cpistack, started, ended):
    benchmark_text = benchmark
    if len(benchmark_text) > 100:
        benchmark_text = benchmark_text[:100] + '__etc'
    run = 'results_{}_{}_{}'.format(BATCH_START, '+'.join(base_configuration), benchmark_text)
    directory = os.path.join(RESULTS_FOLDER, run)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with gzip.open(os.path.join(directory, 'execution.log.gz'), 'w') as f:
        f.write(console_output.encode('utf-8'))
    with open(os.path.join(directory, 'executioninfo.txt'), 'w') as f:
        f.write('started:    {}\n'.format(started.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('ended:      {}\n'.format(ended.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('duration:   {}\n'.format(ended - started))
        f.write('host:       {}\n'.format(platform.node()))
        f.write('tasks:      {}\n'.format(benchmark))
    with open(os.path.join(directory, 'cpi-stack.txt'), 'wb') as f:
        f.write(cpistack)
    for f in ('sim.cfg',
              'sim.info',
              'sim.out',
              'cpi-stack.png',
              'sim.stats.sqlite3'):
        shutil.copy(os.path.join(BENCHMARKS, f), directory)
    for f in ('PeriodicPower.log',
              'PeriodicThermal.log',
              'PeriodicFrequency.log',
              'PeriodicVdd.log',
              'PeriodicCPIStack.log',
              'PeriodicRvalue.log'):
        with open(os.path.join(BENCHMARKS, f), 'rb') as f_in, gzip.open('{}.gz'.format(os.path.join(directory, f)), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    pattern = r"^\d+\.hb.log$" # Heartbeat logs
    for f in os.listdir(BENCHMARKS):
        if not re.match(pattern, f):
            continue
        shutil.copy(os.path.join(BENCHMARKS, f), directory)
    
    for f in os.listdir(BENCHMARKS):
        if 'output.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif 'poses.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif '.264' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif 'app_mapping.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)

    if create_plots is not None:
        create_plots(run)
    else:
        print('plot generation is unavailable ({}); skipped plots for {}'.format(PLOT_IMPORT_ERROR, run))


def run(base_configuration, benchmark, ignore_error=False, perforation_script: str = None, base_cfg_overrides=None):
    print('running {} with configuration {}'.format(benchmark, '+'.join(base_configuration)))
    started = datetime.datetime.now()
    change_base_configuration(base_configuration)
    change_base_configuration_values(base_cfg_overrides)

    prev_run_cleanup()

    benchmark_options = []
    if ENABLE_HEARTBEATS == True:
        benchmark_options.append('enable_heartbeats')
        benchmark_options.append('hb_results_dir=%s' % BENCHMARKS)

    # NOTE: This determines the logging interval! (see issue in forked repo)
    periodicPower = 1000000
    #periodicPower = 250000
    if 'mediumDVFS' in base_configuration:
        periodicPower = 250000
    if 'fastDVFS' in base_configuration:
        periodicPower = 100000

    if not perforation_script:
        perforation_script = 'magic_perforation_rate:' 
   
    args = '-n {number_cores} -c {config} --benchmarks={benchmark} --no-roi --sim-end=last -senergystats:{periodic} -speriodic-power:{periodic}{script}{perforation}{benchmark_options}' \
        .format(number_cores=NUMBER_CORES,
                config=SNIPER_CONFIG,
                benchmark=benchmark,
                periodic=periodicPower,
                script= ''.join([' -s' + s for s in SCRIPTS]),
                perforation=' -s'+perforation_script,
                benchmark_options=''.join([' -B ' + opt for opt in benchmark_options]))
    
    console_output = ''

    print(args)

    run_sniper = os.path.join(BENCHMARKS, 'run-sniper')
    p = subprocess.Popen([run_sniper] + args.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, cwd=BENCHMARKS)
    with p.stdout:
        for line in iter(p.stdout.readline, b''):
            linestr = line.decode('utf-8')
            console_output += linestr
            print(linestr, end='')

    p.wait()

    try:
        cpistack = subprocess.check_output(['python', os.path.join(SNIPER_BASE, 'tools/cpistack.py')], cwd=BENCHMARKS)
    except:
        if ignore_error:
            cpistack = b''
        else:
            raise

    ended = datetime.datetime.now()

    save_output(base_configuration, benchmark, console_output, cpistack, started, ended)

    if p.returncode != 0:
        raise Exception('return code != 0')


def try_run(base_configuration, benchmark, ignore_error=False, base_cfg_overrides=None):
    try:
        run(base_configuration, benchmark, ignore_error=ignore_error, base_cfg_overrides=base_cfg_overrides)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        for i in range(4):
            print('#' * 80)
        #print(e)
        print(traceback.format_exc())
        for i in range(4):
            print('#' * 80)
        input('Please press enter...')


class Infeasible(Exception):
    pass


def get_instance(benchmark, parallelism, input_set='small'):
    threads = {
        'parsec-blackscholes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-bodytrack': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-canneal': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-dedup': [4, 7, 10, 13, 16],
        'parsec-fluidanimate': [2, 3, 0, 5, 0, 0, 0, 9],
        'parsec-streamcluster': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-swaptions': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-x264': [1, 3, 4, 5, 6, 7, 8, 9],
        'splash2-barnes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-cholesky': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-fft': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-fmm': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-lu.cont': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-lu.ncont': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-ocean.cont': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-ocean.ncont': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-radiosity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-radix': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-raytrace': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-water.nsq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-water.sp': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],  # other parallelism values run but are suboptimal -> don't allow in the first place
    }
    
    ps = threads[benchmark]
    if parallelism <= 0 or parallelism not in ps:
        raise Infeasible()
    p = ps.index(parallelism) + 1

    if benchmark.startswith('parsec') and not input_set.startswith('sim'):
        input_set = 'sim' + input_set

    return '{}-{}-{}'.format(benchmark, input_set, p)


def get_feasible_parallelisms(benchmark):
    feasible = []
    for p in range(1, 16+1):
        try:
            get_instance(benchmark, p)
            feasible.append(p)
        except Infeasible:
            pass
    return feasible


def get_workload(benchmark, cores, parallelism=None, number_tasks=None, input_set='small'):
    if parallelism is not None:
        number_tasks = math.floor(cores / parallelism)
        return get_workload(benchmark, cores, number_tasks=number_tasks, input_set=input_set)
    elif number_tasks is not None:
        if number_tasks == 0:
            if cores == 0:
                return []
            else:
                raise Infeasible()
        else:
            parallelism = math.ceil(cores / number_tasks)
            for p in reversed(range(1, min(cores, parallelism) + 1)):
                try:
                    b = get_instance(benchmark, p, input_set=input_set)
                    return [b] + get_workload(benchmark, cores - p, number_tasks=number_tasks-1, input_set=input_set)
                except Infeasible:
                    pass
            raise Infeasible()
    else:
        raise Exception('either parallelism or number_tasks needs to be set')


def h1_overrides(profile_file,
                 target_ips,
                 max_temp,
                 thermal_margin,
                 power_budget,
                 power_budget_margin,
                 per_core_power_guard,
                 freeze_master=True,
                 debug=False,
                 migration_epoch=1000000,
                 dvfs_epoch=1000000):
    states = '1.0,2.0,3.0,4.0'
    core_state = '0,1,2,3'
    frequencies = '1000,2000,3000,4000'
    profile_file = os.path.abspath(profile_file)
    objective = 'target_ips_min_power'
    return {
        'scheduler/open/migration/epoch': migration_epoch,
        'scheduler/open/heuristic_h1/objective': objective,
        'scheduler/open/heuristic_h1/power_budget': power_budget,
        'scheduler/open/heuristic_h1/power_budget_margin': power_budget_margin,
        'scheduler/open/heuristic_h1/per_core_power_guard': per_core_power_guard,
        'scheduler/open/heuristic_h1/max_temp': max_temp,
        'scheduler/open/heuristic_h1/thermal_margin': thermal_margin,
        'scheduler/open/heuristic_h1/num_states': 4,
        'scheduler/open/heuristic_h1/state_value': states,
        'scheduler/open/heuristic_h1/core_state': core_state,
        'scheduler/open/heuristic_h1/target_ips': target_ips,
        'scheduler/open/heuristic_h1/profile_file': profile_file,
        'scheduler/open/heuristic_h1/freeze_master': freeze_master,
        'scheduler/open/heuristic_h1/debug': debug,
        'scheduler/open/dvfs/dvfs_epoch': dvfs_epoch,
        'scheduler/open/dvfs/fixed_states/frequency': frequencies,
        'scheduler/open/dvfs/heuristic_h1/objective': objective,
        'scheduler/open/dvfs/heuristic_h1/power_budget': power_budget,
        'scheduler/open/dvfs/heuristic_h1/power_budget_margin': power_budget_margin,
        'scheduler/open/dvfs/heuristic_h1/per_core_power_guard': per_core_power_guard,
        'scheduler/open/dvfs/heuristic_h1/max_temp': max_temp,
        'scheduler/open/dvfs/heuristic_h1/thermal_margin': thermal_margin,
        'scheduler/open/dvfs/heuristic_h1/num_states': 4,
        'scheduler/open/dvfs/heuristic_h1/state_value': states,
        'scheduler/open/dvfs/heuristic_h1/frequency': frequencies,
        'scheduler/open/dvfs/heuristic_h1/target_ips': target_ips,
        'scheduler/open/dvfs/heuristic_h1/profile_file': profile_file,
        'scheduler/open/dvfs/heuristic_h1/freeze_master': freeze_master,
        'scheduler/open/dvfs/heuristic_h1/debug': debug,
    }


def h1_experiment(benchmark='parsec-blackscholes',
                  parallelism=4,
                  input_set='simsmall',
                  profile_file=None,
                  target_ips=10.0,
                  max_temp=90.0,
                  thermal_margin=3.0,
                  power_budget=0.0,
                  power_budget_margin=1.0,
                  per_core_power_guard=0.0,
                  freeze_master=True,
                  include_maxfreq=False,
                  debug=False,
                  ignore_error=False):
    if profile_file is None:
        profile_file = os.path.join(SNIPER_BASE, 'common/scheduler/policies/heuristic_h1_profiles.example.tsv')
        print('[H1 experiment] using example profile file: {}'.format(profile_file))
        print('[H1 experiment] replace --profile-file with your measured profile for real results.')

    workload = get_instance(benchmark, parallelism, input_set=input_set)
    overrides = h1_overrides(profile_file,
                             target_ips,
                             max_temp,
                             thermal_margin,
                             power_budget,
                             power_budget_margin,
                             per_core_power_guard,
                             freeze_master=freeze_master,
                             debug=debug)
    dvfs_overrides = dict(overrides)
    dvfs_overrides['scheduler/open/migration/logic'] = 'off'

    if include_maxfreq:
        run(['4.0GHz', 'maxFreq', 'slowDVFS'], workload, ignore_error=ignore_error)

    # Static heterogeneous fixed-frequency cores, no migration. This isolates the
    # value of H1's dynamic thread movement.
    run(['fixedStates', 'slowDVFS'], workload, ignore_error=ignore_error, base_cfg_overrides=overrides)

    # H1 dynamic mapping over fixed-VF cores: cores stay at 1/2/3/4 GHz, threads migrate.
    run(['heuristicH1', 'fixedStates', 'slowDVFS'], workload, ignore_error=ignore_error, base_cfg_overrides=overrides)

    # H1 per-core DVFS: threads stay pinned, and H1 directly chooses per-core frequencies.
    run(['heuristicH1DVFS', 'slowDVFS'], workload, ignore_error=ignore_error, base_cfg_overrides=dvfs_overrides)


def example():
    for benchmark in (
                      'parsec-blackscholes',
                      #'parsec-bodytrack',
                      #'parsec-canneal',
                      #'parsec-dedup',
                      #'parsec-ferret'
                      #'parsec-fluidanimate',
                      #'parsec-streamcluster',
                      #'parsec-swaptions',
                      #'parsec-x264',
                      #'splash2-barnes',
                      #'splash2-fmm',
                      #'splash2-ocean.cont',
                      #'splash2-ocean.ncont',
                      #'splash2-radiosity',
                      #'splash2-raytrace',
                      #'splash2-water.nsq',
                      #'splash2-water.sp',
                      #'splash2-cholesky',
                      #'splash2-fft',
                      #'splash2-lu.cont',
                      #'splash2-lu.ncont',
                      #'splash2-radix',
                      ):

        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        max_parallelism = get_feasible_parallelisms(benchmark)[-1]
        for freq in (1, 2):
            #for parallelism in (max_parallelism,):
            for parallelism in (3, ):
                # you can also use try_run instead
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'))

def example_pcgov():
    for benchmark in (
                      'parsec-blackscholes',
                    ):

        for freq in (4, ):
            for parallelism in (3,):
                run(['{:.1f}GHz'.format(freq), 'PCGov', 'slowDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'))

def example_symmetric_perforation():
    for benchmark in (
                      'parsec-blackscholes',
                      #'parsec-bodytrack',
                      #'parsec-streamcluster',
                      #'parsec-swaptions',
                      #'parsec-x264',
                      #'parsec-canneal',
                    ):

        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        max_parallelism = get_feasible_parallelisms(benchmark)[-1]

        perforation_rate = str(50)
        for freq in (4, ):
            for parallelism in (4,):
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'), 
                    perforation_script="magic_perforation_rate:%s" % perforation_rate )

def example_asymmetric_perforation():
    for benchmark in (
                        ("parsec-blackscholes", 1),
                        #("parsec-bodytrack", 6),
                        #("parsec-streamcluster", 2),
                        #("parsec-swaptions", 2),
                        #("parsec-x264", 6),
                        #("parsec-canneal", 3),
                    ):
    
        loop_rates = [  str(i*10) for i in range(benchmark[1]) ]

        min_parallelism = get_feasible_parallelisms(benchmark[0])[0]
        max_parallelism = get_feasible_parallelisms(benchmark[0])[-1]
        for freq in (4, ):
            for parallelism in (4,):
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark[0], parallelism, input_set='simsmall'), 
                    perforation_script='magic_perforation_rate:%s' % ','.join(loop_rates))


def multi_program():
    # In this example, two instances of blackscholes will be scheduled.
    # By setting the scheduler/open/arrivalRate base.cfg parameter to 2, the
    # tasks can be set to arrive at the same time.

    input_set = 'simsmall'
    base_configuration = ['4.0GHz', "maxFreq"]
    benchmark_set = (
        'parsec-blackscholes',
        'parsec-x264',
    )

    if ENABLE_HEARTBEATS == True:
        base_configuration.append('hb_enabled')

    benchmarks = ''
    for i, benchmark in enumerate(benchmark_set):
        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        if i != 0:
            benchmarks = benchmarks + ',' + get_instance(benchmark, min_parallelism, input_set)
        else:
            benchmarks = benchmarks + get_instance(benchmark, min_parallelism, input_set)

    run(base_configuration, benchmarks)

    
def test_static_power():
    run(['4.0GHz', 'testStaticPower', 'slowDVFS'], get_instance('parsec-blackscholes', 3, input_set='simsmall'))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'h1':
        parser = argparse.ArgumentParser(description='Run the 4-core H1 target-IPS/min-power fixed-VF and per-core DVFS experiments.')
        parser.add_argument('--benchmark', default='parsec-blackscholes')
        parser.add_argument('--parallelism', type=int, default=4)
        parser.add_argument('--input-set', default='simsmall')
        parser.add_argument('--profile-file', default=None)
        parser.add_argument('--target-ips', type=float, default=10.0,
                            help='Predicted total IPS target for target_ips_min_power.')
        parser.add_argument('--power-budget', type=float, default=0.0,
                            help='Backward-compatible total predicted power budget for old power_budget_max_ips experiments.')
        parser.add_argument('--power-budget-margin', type=float, default=1.0,
                            help='Multiplier applied to --power-budget before scheduling.')
        parser.add_argument('--per-core-power-guard', type=float, default=0.0,
                            help='Optional per-core predicted power cap; 0 disables it.')
        parser.add_argument('--freeze-master', dest='freeze_master', action='store_true', default=True,
                            help='Keep task-local thread 0 out of H1 migration and DVFS optimization.')
        parser.add_argument('--no-freeze-master', dest='freeze_master', action='store_false',
                            help='Restore legacy behavior where task-local thread 0 participates in H1.')
        parser.add_argument('--max-temp', type=float, default=90.0)
        parser.add_argument('--thermal-margin', type=float, default=3.0)
        parser.add_argument('--include-maxfreq', action='store_true',
                            help='Also run the all-cores max-frequency baseline.')
        parser.add_argument('--debug-h1', action='store_true',
                            help='Enable verbose H1 prediction and migration logs.')
        parser.add_argument('--ignore-error', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        h1_experiment(benchmark=args.benchmark,
                      parallelism=args.parallelism,
                      input_set=args.input_set,
                      profile_file=args.profile_file,
                      target_ips=args.target_ips,
                      max_temp=args.max_temp,
                      thermal_margin=args.thermal_margin,
                      power_budget=args.power_budget,
                      power_budget_margin=args.power_budget_margin,
                      per_core_power_guard=args.per_core_power_guard,
                      freeze_master=args.freeze_master,
                      include_maxfreq=args.include_maxfreq,
                      debug=args.debug_h1,
                      ignore_error=args.ignore_error)
        return


    # example()
    # test_static_power()
    # multi_program()

    # example_symmetric_perforation()
    # example_asymmetric_perforation()
    # example_pcgov()
    
if __name__ == '__main__':
    main()
