#!/usr/bin/env python3
"""
Script to populate blackscholes_expr.json with all available paths from the results folder.
"""

import os
import json

# Configuration
RESULTS_DIR = "../results"
OUTPUT_JSON = "blackscholes_expr.json"
EXPERIMENT_NAME = "migrationSota-3-core"
OUTPUT_DIR = "./txt_out/blackscholes/"

def get_all_result_paths(results_dir):
    """
    Get all result directory paths from the results folder.
    Returns a list of absolute paths sorted by timestamp.
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return []
    
    # Get all directories in the results folder
    entries = os.listdir(results_dir)
    result_dirs = [
        os.path.abspath(os.path.join(results_dir, entry))
        for entry in entries
        if os.path.isdir(os.path.join(results_dir, entry)) and entry.startswith("results_")
    ]
    
    # Sort by directory name (which includes timestamp)
    result_dirs.sort()
    
    return result_dirs

def create_json_structure(result_paths):
    """
    Create the JSON structure with all result paths.
    """
    runs = []
    for idx, path in enumerate(result_paths):
        runs.append({
            "path": path,
            "name": str(idx)
        })
    
    json_data = {
        "experiments": [
            {
                "name": EXPERIMENT_NAME,
                "output_dir": OUTPUT_DIR,
                "runs": runs
            }
        ]
    }
    
    return json_data

def save_json(data, output_file):
    """
    Save the JSON data to file with proper formatting.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully saved {len(data['experiments'][0]['runs'])} paths to {output_file}")

def main():
    print(f"Scanning results directory: {RESULTS_DIR}")
    
    # Get all result paths
    result_paths = get_all_result_paths(RESULTS_DIR)
    
    if not result_paths:
        print("No result directories found!")
        return
    
    print(f"Found {len(result_paths)} result directories:")
    for idx, path in enumerate(result_paths):
        print(f"  [{idx}] {os.path.basename(path)}")
    
    # Create JSON structure
    json_data = create_json_structure(result_paths)
    
    # Save to file
    save_json(json_data, OUTPUT_JSON)
    
    print("\nDone!")

if __name__ == "__main__":
    main()