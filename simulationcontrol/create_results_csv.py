import json
import pandas as pd
import os
import re
from pathlib import Path
from glob import glob

def extract_info_from_path(path):
    """Extract governor, tasks, and benchmark type from result path"""
    basename = os.path.basename(path)
    
    # Extract governor from pattern: results_DATE_FREQ+GOVERNOR+...
    # Pattern: results_YYYY-MM-DD_HH.MM_FREQ+GOVERNOR+...
    governor = None
    match = re.search(r'\d+\.\d+GHz\+([^+]+)', basename)
    if match:
        governor = match.group(1)
    
    # Extract tasks (e.g., parsec-blackscholes-simsmall-2)
    # Look for pattern after the last underscore
    parts = basename.split('_')
    tasks = parts[-1] if parts else basename
    
    # Determine benchmark type
    benchmark_type = 'multi-program' if ',' in tasks else 'single-program'
    
    return governor, benchmark_type, tasks

def read_metric_value(file_path, run_name):
    """Read a specific metric value for a run from a txt file"""
    try:
        df = pd.read_csv(file_path, sep=r'\s+')
        # Convert Run column to string for comparison
        df['Run'] = df['Run'].astype(str)
        row = df[df['Run'] == str(run_name)]
        if not row.empty:
            return row.iloc[0].to_dict()
        else:
            print(f"  Warning: Run '{run_name}' not found in {file_path}")
        return None
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None

def create_combined_csv(json_files, output_csv='experiment_results.csv'):
    """Create a single CSV file with all experiments and metrics"""
    
    all_rows = []
    run_id = 1
    
    for json_file in json_files:
        print(f"\nProcessing {json_file}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for experiment in data['experiments']:
            exp_name = experiment['name']
            output_dir = experiment['output_dir']
            runs = experiment['runs']
            
            # Find metric files in output directory
            metric_files = {}
            for metric_name in ['temps', 'power', 'energy', 'resp-times', 'cpi', 'ips']:
                files = glob(f"{output_dir}/{metric_name}-*.txt")
                if files:
                    metric_key = metric_name.replace('-', '_')
                    metric_files[metric_key] = files[0]
                    print(f"  Found {metric_name}: {files[0]}")
            
            for run in runs:
                row_data = {
                    'run_id': run_id,
                }
                
                # Extract info from path
                governor, benchmark_type, tasks = extract_info_from_path(run['path'])
                row_data['governor'] = governor
                row_data['benchmark_type'] = benchmark_type
                row_data['tasks'] = tasks
                
                run_name = run['name']
                
                # Read metrics from each file
                for metric_name, metric_file in metric_files.items():
                    if os.path.exists(metric_file):
                        metric_data = read_metric_value(metric_file, run_name)
                        if metric_data:
                            if metric_name == 'resp_times':
                                row_data['avg_response_time_ms'] = metric_data.get('resp_time', 0) / 1e6  # Convert ns to ms
                            
                            elif metric_name == 'temps':
                                # Per-core temperatures
                                for i in range(4):
                                    row_data[f'temp_C{i}'] = metric_data.get(f'C_{i}', 0)
                                row_data['avg_temp'] = metric_data.get('total_avg', 0)
                            
                            elif metric_name == 'energy':
                                # Per-core energy
                                for i in range(4):
                                    row_data[f'energy_C{i}_J'] = metric_data.get(f'C_{i}', 0)
                                row_data['total_energy_J'] = metric_data.get('total_energy', 0)
                            
                            elif metric_name == 'power':
                                # Per-core power
                                for i in range(4):
                                    row_data[f'power_C{i}_W'] = metric_data.get(f'C_{i}', 0)
                                row_data['total_power_W'] = metric_data.get('total_sum', 0)
                            
                            elif metric_name == 'cpi':
                                # Per-core CPI
                                core_cpi_sum = 0
                                for i in range(4):
                                    cpi_value = metric_data.get(f'C_{i}', 0)
                                    row_data[f'cpi_C{i}'] = cpi_value
                                    core_cpi_sum += cpi_value
                                # Calculate average correctly by dividing by 4 (all cores)
                                row_data['avg_cpi'] = core_cpi_sum / 4
                            
                            elif metric_name == 'ips':
                                # Per-core IPS
                                core_ips_sum = 0
                                for i in range(4):
                                    ips_value = metric_data.get(f'C_{i}', 0)
                                    row_data[f'ips_C{i}_GIPS'] = ips_value
                                    core_ips_sum += ips_value
                                # Calculate average correctly by dividing by 4 (all cores)
                                row_data['avg_ips_GIPS'] = round(core_ips_sum / 4, 2)
                                row_data['total_ips_GIPS'] = round(core_ips_sum, 2)
                
                all_rows.append(row_data)
                run_id += 1
    
    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    
    # Reorder columns for better readability
    base_cols = ['run_id', 'governor', 'benchmark_type', 'tasks']
    
    # Group metric columns by type
    temp_cols = [f'temp_C{i}' for i in range(4)] + ['avg_temp']
    power_cols = [f'power_C{i}_W' for i in range(4)] + ['total_power_W']
    energy_cols = [f'energy_C{i}_J' for i in range(4)] + ['total_energy_J']
    cpi_cols = [f'cpi_C{i}' for i in range(4)] + ['avg_cpi']
    ips_cols = [f'ips_C{i}_GIPS' for i in range(4)] + ['avg_ips_GIPS', 'total_ips_GIPS']
    resp_cols = ['avg_response_time_ms']
    
    # Combine in logical order
    ordered_cols = base_cols + resp_cols + temp_cols + power_cols + energy_cols + cpi_cols + ips_cols
    
    # Only include columns that exist
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]
    
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Created {output_csv}")
    print(f"  Total runs: {len(df)}")
    print(f"\nColumns ({len(df.columns)}): {', '.join(df.columns.tolist())}")
    print(f"\nPreview:")
    print(df.head(5).to_string(index=False))
    
    return df

if __name__ == '__main__':
    json_files = [
        'blackscholes_expr.json',
        'streamcluster_expr.json'
    ]
    
    print("=" * 80)
    print("Creating combined experiment results CSV")
    print("=" * 80)
    
    df = create_combined_csv(json_files, 'experiment_results.csv')
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)