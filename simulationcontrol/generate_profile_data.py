import re
import gzip
import os
import pandas as pd
import io
from resultlib import get_active_cores
from resultlib import get_tasks
from resultlib import get_config
from resultlib import get_ips_traces
from resultlib import get_cpi_traces



core_names =["C_0","C_1","C_2","C_3"]
num_cores = 4

def _open_file(run, filename):
    full_filename = os.path.join(run, filename)
    if os.path.exists(full_filename):
        return open(full_filename, 'r', encoding="utf-8")

    gzip_filename = '{}.gz'.format(full_filename)
    if os.path.exists(gzip_filename):
        return io.TextIOWrapper(gzip.open(gzip_filename, 'r'), encoding="utf-8")
    
    raise Exception('file does not exist')

def get_file(run, filename):
    full_filename = os.path.join(run, filename)
    if os.path.exists(full_filename):
        return full_filename

    gzip_filename = '{}.gz'.format(full_filename)
    if os.path.exists(gzip_filename):
        return gzip_filename

    raise Exception(filename+' file does not exist')

def get_num_cores(df):
    core_list = df.columns.tolist()
    core_numbers = [int(re.search(r'C_(\d+)', core).group(1)) for core in core_list if re.search(r'C_(\d+)', core)]
    max_core = max(core_numbers)
    return max_core + 1

def get_core_names(df):
    return ['C_{}'.format(i) for i in range(get_num_cores(df))]

def get_core_aggregate(df, core_level=False, atype='sum'):
    cores = get_core_names(df)

    # Core level, return list with one df with aggregated values.
    if core_level:
        core_df = pd.DataFrame()
        for core in cores:
            if atype == 'max':
                core_df[core] = df.filter(regex='{}_*'.format(core)).max(axis=1)
            elif atype == 'min':
                core_df[core] = df.filter(regex='{}_*'.format(core)).min(axis=1)
            else:
                core_df[core] = df.filter(regex='{}_*'.format(core)).sum(axis=1)

        return [("-Cores", core_df)]

    # Subcomponents, return list with df for every core with subcomp data.
    dfs = []
    for core in cores:
        dfs.append(("-" + core, df.filter(regex='{}_*'.format(core))))
    return dfs

def get_average_cpi(run, active_cores):
    traces = get_cpi_traces(run)
    worker_avg =0
    count =0
    for core, trace in enumerate(traces):
        if(core in active_cores):
            valid_trace = [value for value in trace if value is not None]
            if len(valid_trace) > 0:
                tracelen = len(trace)
                sum=0
                for i in trace:
                    if i is not None:
                        sum+=i
                    else:
                        tracelen-=1
                count+=1
                worker_avg +=sum/tracelen
    return float(worker_avg/count)

def get_average_ips(run, active_cores):
    traces = get_ips_traces(run)
    worker_avg =0
    count =0
    for core, trace in enumerate(traces):
        if(core in active_cores):
            valid_trace = [value for value in trace if value is not None]
            if len(valid_trace) > 0:
                tracelen = len(trace)
                sum=0
                for i in trace:
                    if i > 1000:
                        sum+=i
                    else:
                        tracelen-=1

                count+=1
                worker_avg +=sum/tracelen
    return float(worker_avg/count)/1e9

def get_average_temp(run, active_cores):
    filename = get_file(run, 'PeriodicThermal.log')
    df_all = pd.read_csv(filename, delim_whitespace=True)
    dfs = get_core_aggregate(df_all, True, "max")
    for label, df_core in dfs:
        all_core_vals = pd.Series([])
        for core in active_cores:
            val = df_core[core_names[core]].mean()
            all_core_vals[len(all_core_vals)]=(val)
    return all_core_vals.mean()

def get_average_power(run, active_cores):
    filename = get_file(run, 'PeriodicPower.log')
    df_all = pd.read_csv(filename, delim_whitespace=True)
    dfs = get_core_aggregate(df_all, True, "sum")
    for label, df_core in dfs:
        all_core_vals = pd.Series([])
        for core in active_cores:
            val = df_core[core_names[core]].mean()
            all_core_vals[len(all_core_vals)]=(val)
    return all_core_vals.mean()

if __name__ == '__main__':
    runs = []
    dir = "/var/scratch/eeec2607/profiled_data/results/"
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        runs.append(filename)
    runs.sort()
    f = open("profile.txt", "w")
    f.write("name\tstate\tips\tcpi\ttemp\tpower\n")
    for run in runs:
        print(dir+run)
        active_cores = get_active_cores(dir+run)
        active_cores.pop(0) # remove main thread

        name = get_tasks(run)
        state = float(get_config(run).split('+')[0][0:3])
        power = get_average_power(dir+run,active_cores)
        temp = get_average_temp(dir+run,active_cores)
        ips = get_average_ips(dir+run,active_cores)
        cpi = get_average_cpi(dir+run,active_cores)
        f.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(name,state,ips,cpi,temp,power))

    

