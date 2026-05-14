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
from resultlib.plot import create_plots

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


def format_base_cfg_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    return '"{}"'.format(str(value).replace('"', '\\"'))


def apply_base_cfg_overrides(base_cfg_overrides):
    if not base_cfg_overrides:
        return

    base_cfg = os.path.join(SNIPER_BASE, 'config/base.cfg')
    with open(base_cfg, 'r') as f:
        lines = f.read().splitlines()

    current_section = ''
    updated = set()
    output = []

    for line in lines:
        section_match = re.match(r'\s*\[([^\]]+)\]\s*$', line)
        if section_match:
            current_section = section_match.group(1)
            output.append(line)
            continue

        stripped = line.lstrip()
        key_match = re.match(r'([A-Za-z_][A-Za-z_0-9]*)\s*=', stripped)
        if key_match and not stripped.startswith('#') and current_section:
            full_key = '{}/{}'.format(current_section, key_match.group(1))
            if full_key in base_cfg_overrides:
                comment_match = re.search(r'(\s+#.*)$', line)
                trailing_comment = comment_match.group(1) if comment_match else ''
                line = '{} = {}{}'.format(
                    key_match.group(1),
                    format_base_cfg_value(base_cfg_overrides[full_key]),
                    trailing_comment,
                )
                updated.add(full_key)

        output.append(line)

    missing = sorted(set(base_cfg_overrides) - updated)
    if missing:
        raise KeyError('base.cfg override key(s) not found: {}'.format(', '.join(missing)))

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

    create_plots(run)


def run(base_configuration, benchmark, ignore_error=False, perforation_script: str = None, base_cfg_overrides=None):
    print('running {} with configuration {}'.format(benchmark, '+'.join(base_configuration)))
    started = datetime.datetime.now()
    change_base_configuration(base_configuration)
    apply_base_cfg_overrides(base_cfg_overrides)

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


def try_run(base_configuration, benchmark, ignore_error=False):
    try:
        run(base_configuration, benchmark, ignore_error=ignore_error)
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


def dyn_thread_mapping_run(args):
    base_cfg_overrides = {}
    thermal_sample_requested = bool(
        args.thermal_sample_file
        or args.thermal_sample_debug
        or args.thermal_sample_random
    )
    base_cfg_overrides['scheduler/open/thermal_sampler/enabled'] = thermal_sample_requested
    base_cfg_overrides['scheduler/open/thermal_sampler/debug'] = bool(args.thermal_sample_debug)
    base_cfg_overrides['scheduler/open/thermal_sampler/exploration_enabled'] = bool(args.thermal_sample_random)
    base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/thermal_model_debug'] = bool(args.thermal_model_debug)
    base_cfg_overrides['scheduler/open/dvfs/reserved_cores_are_active'] = bool(args.reserved_cores_active)
    base_cfg_overrides['scheduler/open/migration/logic'] = 'off' if args.disable_migration else 'DynThreadMapping'

    if args.profile_file:
        base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/profile_path'] = args.profile_file
    if args.thermal_model_file:
        base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/thermal_model_path'] = args.thermal_model_file
    if args.prediction_temperature_bar is not None:
        base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/prediction_temperature_bar'] = args.prediction_temperature_bar
    if args.prediction_safety_margin is not None:
        base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/prediction_safety_margin'] = args.prediction_safety_margin
    if args.migration_utilization_delta_threshold is not None:
        base_cfg_overrides['scheduler/open/migration/DynThreadMapping/utilization_delta_threshold'] = args.migration_utilization_delta_threshold
    if args.master_migration_temp_delta is not None:
        base_cfg_overrides['scheduler/open/migration/DynThreadMapping/master_temperature_delta_threshold'] = args.master_migration_temp_delta
    if args.master_migration_cooldown_ns is not None:
        base_cfg_overrides['scheduler/open/migration/DynThreadMapping/master_cooldown_ns'] = args.master_migration_cooldown_ns
    if args.migration_epoch_ns is not None:
        base_cfg_overrides['scheduler/open/migration/epoch'] = args.migration_epoch_ns
    if args.dvfs_epoch_ns is not None:
        base_cfg_overrides['scheduler/open/dvfs/dvfs_epoch'] = args.dvfs_epoch_ns
    if args.thermal_sample_epoch_ns is not None:
        base_cfg_overrides['scheduler/open/thermal_sampler/epoch'] = args.thermal_sample_epoch_ns
    if args.thermal_sample_file:
        base_cfg_overrides['scheduler/open/thermal_sampler/path'] = args.thermal_sample_file
    if args.thermal_sample_min_temp is not None:
        base_cfg_overrides['scheduler/open/thermal_sampler/target_min_temperature'] = args.thermal_sample_min_temp
    if args.thermal_sample_max_temp is not None:
        base_cfg_overrides['scheduler/open/thermal_sampler/target_max_temperature'] = args.thermal_sample_max_temp
    if args.thermal_sample_migration_probability is not None:
        base_cfg_overrides['scheduler/open/thermal_sampler/migration_probability'] = args.thermal_sample_migration_probability
    if args.thermal_sample_random_seed is not None:
        base_cfg_overrides['scheduler/open/thermal_sampler/random_seed'] = args.thermal_sample_random_seed

    base_configuration = [
        '{:.1f}GHz'.format(args.frequency),
        'DynThreadMapping',
        args.dvfs_speed,
    ]
    benchmark = get_instance(args.benchmark, args.parallelism, input_set=args.input_set)
    run(
        base_configuration,
        benchmark,
        ignore_error=args.ignore_error,
        base_cfg_overrides=base_cfg_overrides,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h1', action='store_true',
                        help='Alias for the default DynThreadMapping experiment.')
    parser.add_argument('--dyn-thread-mapping', action='store_true',
                        help='Run the default DynThreadMapping experiment.')
    parser.add_argument('--thermal-model-file', default=None,
                        help='C++ text model path for scheduler/open/dvfs/DynThreadMapping/thermal_model_path.')
    parser.add_argument('--prediction-temperature-bar', type=float, default=None,
                        help='Upper bound for predicted candidate temperature before DynThreadMapping will lower frequency.')
    parser.add_argument('--prediction-safety-margin', type=float, default=None,
                        help='Conservative margin subtracted from prediction-temperature-bar when testing candidate safety.')
    parser.add_argument('--migration-utilization-delta-threshold', type=float, default=None,
                        help='Minimum high-vs-low core utilization delta required before DynThreadMapping migration moves threads.')
    parser.add_argument('--master-migration-temp-delta', type=float, default=None,
                        help='Minimum current-vs-target core temperature delta required for master-only migration.')
    parser.add_argument('--master-migration-cooldown-ns', type=int, default=None,
                        help='Cooldown in ns between master-only migrations.')
    parser.add_argument('--migration-epoch-ns', type=int, default=None,
                        help='Migration policy epoch in ns.')
    parser.add_argument('--dvfs-epoch-ns', type=int, default=None,
                        help='DynThreadMapping DVFS policy epoch in ns.')
    parser.add_argument('--thermal-sample-file', default=None,
                        help='Enable scheduler thermal sampling and write interval CSV rows to this path.')
    parser.add_argument('--thermal-sample-epoch-ns', type=int, default=None,
                        help='Thermal sampler interval in ns; should match the policy epoch used for training.')
    parser.add_argument('--thermal-sample-debug', action='store_true',
                        help='Print predicted-vs-actual temperature and old/new frequency for each thermal sample interval.')
    parser.add_argument('--thermal-sample-random', action='store_true',
                        help='Randomize migration and frequencies while thermal sampling.')
    parser.add_argument('--thermal-sample-min-temp', type=float, default=None,
                        help='Lower target temperature for random thermal sampling.')
    parser.add_argument('--thermal-sample-max-temp', type=float, default=None,
                        help='Upper target temperature for random thermal sampling.')
    parser.add_argument('--thermal-sample-migration-probability', type=float, default=None,
                        help='Probability of applying a random migration permutation in thermal sample random mode.')
    parser.add_argument('--thermal-sample-random-seed', type=int, default=None,
                        help='Random seed for thermal sample random mode.')
    parser.add_argument('--profile-file', default=None,
                        help='Profile file path for NeighborPrediction.')
    parser.add_argument('--thermal-model-debug', action='store_true',
                        help='Print every ML temperature prediction.')
    parser.add_argument('--reserved-cores-active', action='store_true',
                        help='Treat task-reserved cores as active for DynThreadMapping DVFS.')
    parser.add_argument('--disable-migration', action='store_true',
                        help='Disable DynThreadMapping migration so DVFS prediction runs every epoch.')
    parser.add_argument('--benchmark', default='parsec-blackscholes')
    parser.add_argument('--parallelism', type=int, default=3)
    parser.add_argument('--input-set', default='simsmall')
    parser.add_argument('--frequency', type=float, default=3.0)
    parser.add_argument('--dvfs-speed', default='slowDVFS')
    parser.add_argument('--ignore-error', action='store_true')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        example()
        # test_static_power()
        # multi_program()
        return

    if (args.h1 or args.dyn_thread_mapping or args.thermal_model_file or args.profile_file
            or args.prediction_temperature_bar is not None
            or args.prediction_safety_margin is not None
            or args.migration_utilization_delta_threshold is not None
            or args.master_migration_temp_delta is not None
            or args.master_migration_cooldown_ns is not None
            or args.migration_epoch_ns is not None
            or args.dvfs_epoch_ns is not None
            or args.thermal_sample_file
            or args.thermal_sample_epoch_ns is not None
            or args.thermal_sample_debug
            or args.thermal_sample_random
            or args.thermal_sample_min_temp is not None
            or args.thermal_sample_max_temp is not None
            or args.thermal_sample_migration_probability is not None
            or args.thermal_sample_random_seed is not None):
        dyn_thread_mapping_run(args)
        return

    example()

    # example_symmetric_perforation()
    # example_asymmetric_perforation()
    # example_pcgov()
    
if __name__ == '__main__':
    main()
