import re
import gzip
import os
import pandas as pd
import io
import json
from resultlib import get_active_cores
from resultlib import get_ips_traces

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

def get_average_response_time(run):
    with _open_file(run, 'execution.log') as f:
        for line in f:
            m = re.search(r'Average Response Time \(ns\)\s+:\s+(\d+)', line)
            if m is not None:
                return int(m.group(1))

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

def create_cpi_stack_data(runs,out_filename):
    f = open(out_filename, "w")
    f.write("Run "+core_names[0]+" "+core_names[1]+" "+core_names[2]+" "+core_names[3]+"\n")
    cpi_core_names =["Core 0","Core 1","Core 2","Core 3"]

    for run in runs:
        filename = get_file(run["path"], "cpi-stack.txt")
        df_all = pd.read_csv(filename, sep=r"\s{2,}", engine="python")
        df_all = df_all.set_index("CPI")
        f.write(run["name"]+" ")
        for name in cpi_core_names:
            try:
                base = df_all.loc["base",name]
            except KeyError:
                base = 0
            try:
                depend_int = df_all.loc["depend-int",name]
            except KeyError:
                depend_int = 0 
            try:
                depend_fp = df_all.loc["depend-fp",name]
            except KeyError:
                depend_fp = 0
            try:
                issue = df_all.loc["issue",name]
            except KeyError:
                issue = 0
            try:
                mem_l3 = df_all.loc["mem-l3",name]
            except KeyError:
                mem_l3 = 0

            compute_avg = (mem_l3)/df_all.loc["total",name]
            f.write("{:.2f} ".format(compute_avg))
        f.write("\n")

def get_cpi_stack_part_trace(run, part='total'):
    traces = []

    trace_values = []
    with _open_file(run, 'PeriodicCPIStack.log') as f:
        f.readline()
        for line in f:
            if line.startswith(part + '\t'):
                items = line.split()[1:]
                if items == ['-']:
                    ps = [0] * 4
                else:
                    ps = [float(value) for value in items]
                trace_values.append(ps)

    return list(zip(*trace_values))

def get_cpi_traces(run, raw=False):
    traces = list(map(list, get_cpi_stack_part_trace(run, 'total')))
    if not raw:
        w = 2
        for trace in traces:
            drop = [i for i in range(len(trace)) if any(t > 20 for t in trace[max(i-w,0):min(i+w,len(trace))])]
            for i in drop:
                trace[i] = None
    return traces

def create_cpi_data_txts(runs,out_filename):
    f = open(out_filename, "w")
    f.write("Run "+core_names[0]+" "+core_names[1]+" "+core_names[2]+" "+core_names[3]+" avg_workers\n")
    for run in runs:
        filename = get_file(run["path"], "cpi-stack.txt")

        traces = get_cpi_traces(run["path"])
        worker_avg =0
        count =0
        f.write(run["name"]+" ")
        for core, trace in enumerate(traces):
            valid_trace = [value for value in trace if value is not None]
            if len(valid_trace) > 0:
                tracelen = len(trace)
                sum=0
                for i in trace:
                    if i is not None:
                        sum+=i
                    else:
                        tracelen-=1

                f.write("{:.2f} ".format(sum/tracelen))
                if(core>0):
                    count+=1
                    worker_avg +=sum/tracelen
            else:
                f.write("0.0 ")
        f.write("{:.2f} ".format(worker_avg/count))
        f.write("\n")
        


# Take a set of runs for an experiment and output a text file with data from the log file for each run
def create_experiment_data_txts(runs,log, out_filename,core_level=False, atype='sum',all_core_metric='avg'):
    # create output file
    f = open(out_filename, "w")
    f.write("Run "+core_names[0]+" "+core_names[1]+" "+core_names[2]+" "+core_names[3]+" total_"+all_core_metric+"\n")
    for run in runs:
        filename = get_file(run["path"], log)
        df_all = pd.read_csv(filename, delim_whitespace=True)
        dfs = get_core_aggregate(df_all, core_level, atype)
        

        for label, df_core in dfs:
            all_core_vals = pd.Series([])
            f.write(run["name"]+" ")
            for name in core_names:
                val = df_core[name].mean()
                all_core_vals[len(all_core_vals)]=(val)
                f.write("{:.2f} ".format(val))
            
            # write the value used for all cores
            if(all_core_metric=="avg"): # average of all core values
                all_core_vals[len(all_core_vals)]=df_all["L3"].mean()
                f.write("{:.2f} \n".format(all_core_vals.mean()))
            elif(all_core_metric=="sum"): # sum of core values
                f.write("{:.2f} \n".format(all_core_vals.sum()+df_all["L3"].mean()))

# Take set of runs for an experiment and output a text file with response times for each run
def create_resp_times_data_txt(runs,out_filename):
    # create output file
    f = open(out_filename, "w")
    f.write("Run resp_time\n")

    for run in runs:
        rsp_time = get_average_response_time(run["path"])
        f.write(run["name"]+" {}".format(rsp_time)+"\n")

# Take set of runs for an experiment and output a text file with energy consumption for each core and sum of all cores for each run
def create_energy_data_txt(runs,out_filename):
    # create output file
    f = open(out_filename, "w")
    f.write("Run "+core_names[0]+" "+core_names[1]+" "+core_names[2]+" "+core_names[3]+" total_energy\n")

    for run in runs:
        filename = get_file(run["path"], "PeriodicPower.log")
        df_all = pd.read_csv(filename, delim_whitespace=True)
        dfs = get_core_aggregate(df_all, True)

        for label, df_core in dfs:
            all_core_vals = pd.Series([])
            f.write(run["name"]+" ")
            rsp_time = get_average_response_time(run["path"])
            time = rsp_time / 1e9
            for name in core_names:
                dt = 1e-3  # 1 ms in seconds
                power_avg = sum(df_core[name])/len(df_core[name])

                energy_J = power_avg*time
                all_core_vals[len(all_core_vals)]=(energy_J)
                f.write("{:.2f} ".format(energy_J))
            
            l3_energy=df_all["L3"].mean()*time
            f.write("{:.2f} \n".format(all_core_vals.sum()+l3_energy))

def create_ips_data_txts(runs,out_filename):
    f = open(out_filename, "w")
    f.write("Run "+core_names[0]+" "+core_names[1]+" "+core_names[2]+" "+core_names[3]+" sum_workers\n")
    for run in runs:
        f.write(run["name"]+" ")
        active_cores = get_active_cores(run["path"])
        traces = get_ips_traces(run["path"])
        worker_avg =0
        count =0
        for core, trace in enumerate(traces):
            if(core in active_cores):
                valid_trace = [value for value in trace if value is not None]
                if len(valid_trace) > 0:
                    tracelen = len(trace)
                    sum=0
                    for i in trace:
                        if i >0:
                            sum+=i
                        else:
                            tracelen-=1
                    f.write("{:.2f} ".format(float(sum/tracelen)/1e9))
                    if(core>0):
                        count+=1
                        worker_avg +=sum/tracelen
            else:   
                f.write("{:.2f} ".format(0.0))
        avg= float(worker_avg)/1e9
        f.write("{:.2f} ".format(avg))
        f.write("\n")


def create_max_temps(runs,out_filename):
    # create output file
    f = open(out_filename, "w")
    f.write("Run peak_temp\n")
    for run in runs:
        filename = get_file(run["path"], "PeriodicThermal.log")
        df_all = pd.read_csv(filename, delim_whitespace=True)
        dfs = get_core_aggregate(df_all, True, 'max')
        f.write(run["name"])
        peak_temp =0.0
        for label, df_core in dfs:
            for name in core_names:
                val = df_core[name].max()
                if(val>peak_temp):
                    peak_temp = val
            
        f.write(" {:.2f}\n".format(peak_temp))


def create_data(experiments):
    for experiment in experiments["experiments"]:
        runs = experiment["runs"]
        out_path = experiment["output_dir"]
        name = experiment["name"]
        # Create text file for temps
        create_experiment_data_txts(runs,'PeriodicThermal.log',out_path+"temps-"+name+".txt",core_level=True,atype='max',all_core_metric="avg")
        
        #Create text file for power
        create_experiment_data_txts(runs,'PeriodicPower.log',out_path+"power-"+name+".txt",core_level=True,atype='sum',all_core_metric="sum")
        
        #Create text file for response time
        create_resp_times_data_txt(runs,out_path+"resp-times-"+name+".txt")
        
        #Create text file for energy
        create_energy_data_txt(runs,out_path+"energy-"+name+".txt")
        create_cpi_data_txts(runs,out_path+"cpi-"+name+".txt")
        create_ips_data_txts(runs,out_path+"ips-"+name+".txt")
        create_max_temps(runs,out_path+"peak-temp-"+name+".txt")

        #create_cpi_stack_data(runs,out_path+"cpi-"+name+".txt")


if __name__ == '__main__':
    #with open("blackscholes_expr.json") as f:
    #    blackscholes_experiments = json.load(f)
    #create_data(blackscholes_experiments)
    
    with open("streamcluster_expr.json") as f:
        streamcluster_experiments = json.load(f)
    create_data(streamcluster_experiments)
    

