import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for graphs
output_dir = 'graphs'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV
df = pd.read_csv('experiment_results.csv')

# Define metrics to plot
metrics = {
    'avg_response_time_ms': 'Average Response Time (ms)',
    'avg_temp': 'Average Temperature (°C)',
    'total_energy_J': 'Total Energy (J)',
    'total_power_W': 'Total Power (W)',
    'avg_cpi': 'Average CPI',
    'avg_ips_GIPS': 'Average IPS (GIPS)'
}

# Create a shortened task label
def create_label(row):
    tasks = row['tasks']
    if 'blackscholes' in tasks and 'streamcluster' in tasks:
        return 'mixed'
    elif 'blackscholes' in tasks:
        return 'blackscholes'
    elif 'streamcluster' in tasks:
        return 'streamcluster'
    return tasks

df['benchmark'] = df.apply(create_label, axis=1)
df['config'] = df['benchmark'] + '-' + df['benchmark_type'].str.replace('-program', '')

# Create graphs for each metric
for metric_col, metric_name in metrics.items():
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data
    plot_data = df[['config', 'governor', metric_col]].copy()
    
    # Create grouped bar chart
    configs = plot_data['config'].unique()
    governors = plot_data['governor'].unique()
    
    x = range(len(configs))
    width = 0.35
    
    for i, governor in enumerate(governors):
        gov_data = []
        for config in configs:
            val = plot_data[(plot_data['config'] == config) & (plot_data['governor'] == governor)][metric_col]
            gov_data.append(val.values[0] if len(val) > 0 else 0)
        
        offset = width * (i - len(governors)/2 + 0.5)
        bars = ax.bar([xi + offset for xi in x], gov_data, width, label=governor)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/{metric_col}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Created {filename}")
    plt.close()

# Create a comprehensive comparison graph
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (metric_col, metric_name) in enumerate(metrics.items()):
    ax = axes[idx]
    
    plot_data = df[['config', 'governor', metric_col]].copy()
    configs = plot_data['config'].unique()
    governors = plot_data['governor'].unique()
    
    x = range(len(configs))
    width = 0.35
    
    for i, governor in enumerate(governors):
        gov_data = []
        for config in configs:
            val = plot_data[(plot_data['config'] == config) & (plot_data['governor'] == governor)][metric_col]
            gov_data.append(val.values[0] if len(val) > 0 else 0)
        
        offset = width * (i - len(governors)/2 + 0.5)
        ax.bar([xi + offset for xi in x], gov_data, width, label=governor)
    
    ax.set_xlabel('Configuration', fontsize=10, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
    ax.set_title(metric_name, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Complete Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
filename = f"{output_dir}/all_metrics_comparison.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Created {filename}")
plt.close()

# Create per-core comparison for temperature
fig, ax = plt.subplots(figsize=(14, 7))

core_temp_cols = ['temp_C0', 'temp_C1', 'temp_C2', 'temp_C3']
x = range(len(df))
width = 0.15

for i, core_col in enumerate(core_temp_cols):
    offset = width * (i - len(core_temp_cols)/2 + 0.5)
    ax.bar([xi + offset for xi in x], df[core_col], width, label=f'Core {i}')

ax.set_xlabel('Run ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('Per-Core Temperature Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['run_id'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add governor labels
for i, row in df.iterrows():
    ax.text(i, -10, row['governor'][:4], ha='center', fontsize=8, rotation=45)

plt.tight_layout()
filename = f"{output_dir}/per_core_temperature.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Created {filename}")
plt.close()

# Create per-core comparison for power
fig, ax = plt.subplots(figsize=(14, 7))

core_power_cols = ['power_C0_W', 'power_C1_W', 'power_C2_W', 'power_C3_W']

for i, core_col in enumerate(core_power_cols):
    offset = width * (i - len(core_power_cols)/2 + 0.5)
    ax.bar([xi + offset for xi in x], df[core_col], width, label=f'Core {i}')

ax.set_xlabel('Run ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
ax.set_title('Per-Core Power Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['run_id'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, row in df.iterrows():
    ax.text(i, -0.5, row['governor'][:4], ha='center', fontsize=8, rotation=45)

plt.tight_layout()
filename = f"{output_dir}/per_core_power.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Created {filename}")
plt.close()

print("\n" + "="*80)
print("All graphs created successfully!")
print(f"Graphs saved in '{output_dir}/' directory")
print("="*80)
