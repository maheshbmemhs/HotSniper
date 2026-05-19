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

HERE = os.path.dirname(os.path.abspath(__file__))
SNIPER_BASE = os.path.dirname(HERE)
BENCHMARKS = os.path.join(SNIPER_BASE, 'benchmarks')
BATCH_START = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')

THREAD_REQUIREMENTS = {
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
    'splash2-water.sp': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
}


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

    try:
        from resultlib.plot import create_plots
        create_plots(run)
    except ImportError as e:
        print('Skipping plots because plotting dependencies are unavailable: {}'.format(e))


def run(base_configuration, benchmark, ignore_error=False, perforation_script: str = None,
        base_cfg_overrides=None, number_cores=None):
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
        .format(number_cores=number_cores or NUMBER_CORES,
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
    ps = THREAD_REQUIREMENTS[benchmark]
    if parallelism <= 0 or parallelism not in ps:
        raise Infeasible()
    p = ps.index(parallelism) + 1

    if benchmark.startswith('parsec') and not input_set.startswith('sim'):
        input_set = 'sim' + input_set

    return '{}-{}-{}'.format(benchmark, input_set, p)


def split_csv(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def unique(items):
    values = []
    seen = set()
    for item in items:
        if item not in seen:
            values.append(item)
            seen.add(item)
    return values


def parse_parallelisms(value):
    if not value:
        return []
    parallelisms = []
    for item in split_csv(value):
        try:
            parallelisms.append(int(item))
        except ValueError:
            raise ValueError('invalid parallelism "{}"'.format(item))
    return parallelisms


def is_benchmark_instance(spec):
    fields = spec.split('-')
    return len(fields) == 4 and fields[3].isdigit()


def parse_benchmark_instance(instance):
    fields = instance.split('-')
    if len(fields) != 4:
        raise ValueError('invalid benchmark instance "{}"'.format(instance))
    return fields[0], fields[1], fields[2], fields[3]


def get_instance_core_requirement(instance):
    suite, program, input_set, parallelism = parse_benchmark_instance(instance)
    benchmark = '{}-{}'.format(suite, program)
    instance_parallelism = int(parallelism)
    if benchmark not in THREAD_REQUIREMENTS:
        raise ValueError('unknown benchmark "{}" in instance "{}"'.format(benchmark, instance))
    requirements = THREAD_REQUIREMENTS[benchmark]
    if instance_parallelism < 1 or instance_parallelism > len(requirements):
        raise Infeasible()
    core_requirement = requirements[instance_parallelism - 1]
    if core_requirement <= 0:
        raise Infeasible()
    return core_requirement


def build_benchmark_instance(spec, parallelism=None, input_set='small'):
    if ':' in spec:
        spec, spec_parallelism = spec.rsplit(':', 1)
        if parallelism is not None:
            raise ValueError('parallelism provided twice for "{}"'.format(spec))
        try:
            parallelism = int(spec_parallelism)
        except ValueError:
            raise ValueError('invalid parallelism "{}"'.format(spec_parallelism))

    if is_benchmark_instance(spec):
        if parallelism is not None:
            raise ValueError('prebuilt benchmark instance "{}" cannot also use --parallelisms'.format(spec))
        get_instance_core_requirement(spec)
        return spec

    if parallelism is None:
        raise ValueError('missing parallelism for "{}"'.format(spec))
    return get_instance(spec, parallelism, input_set=input_set)


def build_workload(args):
    if args.workload and args.benchmarks:
        raise ValueError('use either --workload or --benchmarks, not both')

    if args.workload:
        specs = split_csv(args.workload)
        default_parallelisms = parse_parallelisms(args.parallelisms)
    elif args.benchmarks:
        specs = split_csv(args.benchmarks)
        default_parallelisms = parse_parallelisms(args.parallelisms)
        if not default_parallelisms:
            default_parallelisms = [args.parallelism]
    else:
        specs = [args.benchmark]
        default_parallelisms = [args.parallelism]

    if not specs:
        raise ValueError('workload is empty')

    if default_parallelisms:
        if len(default_parallelisms) == 1 and len(specs) > 1:
            default_parallelisms = default_parallelisms * len(specs)
        elif len(default_parallelisms) != len(specs):
            raise ValueError('--parallelisms must have one value or match the number of benchmarks')

    instances = []
    for i, spec in enumerate(specs):
        parallelism = default_parallelisms[i] if default_parallelisms else None
        instances.append(build_benchmark_instance(spec, parallelism, input_set=args.input_set))

    total_cores = sum(get_instance_core_requirement(instance) for instance in instances)
    return ','.join(instances), instances, total_cores


def workload_benchmarks(instances):
    benchmarks = []
    seen = set()
    for instance in instances:
        suite, program, input_set, parallelism = parse_benchmark_instance(instance)
        if program in seen:
            continue
        benchmarks.append({
            'suite': suite,
            'program': program,
            'suite_program': '{}-{}'.format(suite, program),
            'instance': instance,
        })
        seen.add(program)
    return benchmarks


def normalize_workload_model_key(key):
    return re.sub(r'[\s,_]+', '+', key.strip().lower())


def parse_thermal_model_files(value):
    mapping = {}
    if not value:
        return mapping
    for item in split_csv(value):
        if '=' not in item:
            raise ValueError('invalid --thermal-model-files entry "{}"; expected benchmark=path'.format(item))
        key, path = item.split('=', 1)
        key = normalize_workload_model_key(key)
        path = path.strip()
        if not key or not path:
            raise ValueError('invalid --thermal-model-files entry "{}"; expected benchmark=path'.format(item))
        mapping[key] = path
    return mapping


def find_thermal_model_mapping(mapping, benchmark):
    for key in (
        benchmark['instance'],
        benchmark['suite_program'],
        benchmark['program'],
    ):
        normalized = normalize_workload_model_key(key)
        if normalized in mapping:
            return mapping[normalized]
    return None


def thermal_model_epoch_label(args):
    if args.thermal_model_suffix:
        return args.thermal_model_suffix

    epoch_ns = args.dvfs_epoch_ns
    if epoch_ns is None:
        if args.dvfs_speed == 'fastDVFS':
            epoch_ns = 100000
        elif args.dvfs_speed == 'mediumDVFS':
            epoch_ns = 250000
        else:
            epoch_ns = 1000000

    epoch_ms = epoch_ns / 1000000.0
    if epoch_ms.is_integer():
        return '{}ms'.format(int(epoch_ms))
    return '{}ms'.format(('{:.6g}'.format(epoch_ms)).replace('.', 'p'))


def path_exists_from_simulationcontrol(path):
    if os.path.isabs(path):
        return os.path.exists(path)
    return os.path.exists(os.path.normpath(os.path.join(HERE, path)))


def infer_thermal_model_file(instances, args):
    suffix = thermal_model_epoch_label(args)
    benchmarks = workload_benchmarks(instances)
    resolved = []
    candidates = []

    for benchmark in benchmarks:
        candidate = os.path.join('..', 'ml_models', '{}_{}.txt'.format(benchmark['program'], suffix))
        candidates.append(candidate)
        if not path_exists_from_simulationcontrol(candidate):
            return None, candidates
        resolved.append('{}={}'.format(benchmark['program'], candidate))

    if len(resolved) == 1:
        return resolved[0].split('=', 1)[1], candidates
    return ','.join(resolved), candidates


def resolve_thermal_model_file(args, instances):
    if args.thermal_model_file and args.thermal_model_files:
        raise ValueError('use either --thermal-model-file or --thermal-model-files, not both')

    if args.thermal_model_file:
        return args.thermal_model_file

    benchmarks = workload_benchmarks(instances)
    if args.thermal_model_files:
        mapping = parse_thermal_model_files(args.thermal_model_files)
        resolved = []
        missing = []
        for benchmark in benchmarks:
            path = find_thermal_model_mapping(mapping, benchmark)
            if path:
                resolved.append('{}={}'.format(benchmark['program'], path))
            else:
                missing.append(benchmark['program'])

        if missing:
            raise ValueError(
                'missing thermal model mapping for benchmark(s): {}. '
                'Use keys like blackscholes=../ml_models/blackscholes_1ms.txt.'.format(
                    ', '.join(missing),
                )
            )

        if len(resolved) == 1:
            return resolved[0].split('=', 1)[1]
        return ','.join(resolved)

    inferred, candidates = infer_thermal_model_file(instances, args)
    if inferred:
        return inferred

    if len(benchmarks) > 1:
        raise ValueError(
            'multi-program workload "{}" needs per-benchmark thermal model files. Tried: {}. '
            'Pass --thermal-model-file or --thermal-model-files.'.format(
                '+'.join(instances),
                ', '.join(candidates),
            )
        )

    return None


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
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'fastDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'))

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
    if args.arrival_rate is not None:
        base_cfg_overrides['scheduler/open/arrivalRate'] = args.arrival_rate
    if args.arrival_interval_ns is not None:
        base_cfg_overrides['scheduler/open/arrivalInterval'] = args.arrival_interval_ns
    if args.arrival_distribution is not None:
        base_cfg_overrides['scheduler/open/distribution'] = args.arrival_distribution

    base_configuration = [
        '{:.1f}GHz'.format(args.frequency),
        'DynThreadMapping',
        args.dvfs_speed,
    ]
    benchmark, workload_instances, workload_cores = build_workload(args)
    thermal_model_file = resolve_thermal_model_file(args, workload_instances)
    if thermal_model_file:
        base_cfg_overrides['scheduler/open/dvfs/DynThreadMapping/thermal_model_path'] = thermal_model_file

    number_cores = args.cores or NUMBER_CORES
    if workload_cores > number_cores:
        raise ValueError('workload requires {} cores, but --cores/config.py NUMBER_CORES is {}'.format(
            workload_cores,
            number_cores,
        ))
    if len(workload_instances) > 1 and args.arrival_rate is None:
        base_cfg_overrides['scheduler/open/arrivalRate'] = len(workload_instances)

    run(
        base_configuration,
        benchmark,
        ignore_error=args.ignore_error,
        base_cfg_overrides=base_cfg_overrides,
        number_cores=number_cores,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h1', action='store_true',
                        help='Alias for the default DynThreadMapping experiment.')
    parser.add_argument('--dyn-thread-mapping', action='store_true',
                        help='Run the default DynThreadMapping experiment.')
    parser.add_argument('--thermal-model-file', default=None,
                        help='C++ text model path for scheduler/open/dvfs/DynThreadMapping/thermal_model_path.')
    parser.add_argument('--thermal-model-files', default=None,
                        help='Comma-separated benchmark=path mappings for per-benchmark thermal models.')
    parser.add_argument('--thermal-model-suffix', default=None,
                        help='Suffix used for automatic thermal model lookup, e.g. 1ms or 0p1ms. Defaults from --dvfs-epoch-ns/--dvfs-speed.')
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
    parser.add_argument('--benchmarks', default=None,
                        help='Comma-separated benchmark names for a multi-program workload.')
    parser.add_argument('--parallelisms', default=None,
                        help='Comma-separated total core requirements matching --benchmarks. A single value is reused for all benchmarks; defaults to --parallelism.')
    parser.add_argument('--workload', default=None,
                        help='Comma-separated workload specs: benchmark:parallelism or prebuilt suite-benchmark-input-nthreads instances.')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of Sniper application cores. Defaults to simulationcontrol/config.py NUMBER_CORES.')
    parser.add_argument('--arrival-rate', type=int, default=None,
                        help='Override scheduler/open/arrivalRate. Multi-program workloads default to all tasks arriving together.')
    parser.add_argument('--arrival-interval-ns', type=int, default=None,
                        help='Override scheduler/open/arrivalInterval.')
    parser.add_argument('--arrival-distribution', choices=('uniform', 'poisson', 'explicit'), default=None,
                        help='Override scheduler/open/distribution.')
    parser.add_argument('--input-set', default='simsmall')
    parser.add_argument('--frequency', type=float, default=3.0)
    parser.add_argument('--dvfs-speed', default='fastDVFS')
    parser.add_argument('--ignore-error', action='store_true')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        example()
        # test_static_power()
        # multi_program()
        return

    if (args.h1 or args.dyn_thread_mapping or args.thermal_model_file or args.thermal_model_files
            or args.thermal_model_suffix or args.profile_file
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
            or args.thermal_sample_random_seed is not None
            or args.workload
            or args.benchmarks
            or args.cores is not None
            or args.arrival_rate is not None
            or args.arrival_interval_ns is not None
            or args.arrival_distribution is not None):
        dyn_thread_mapping_run(args)
        return

    example()

    # example_symmetric_perforation()
    # example_asymmetric_perforation()
    # example_pcgov()
    
if __name__ == '__main__':
    main()
