import subprocess
import sys

# List your files in the exact order they need to run
scripts = [
    'code/get_data/cpl_team_stats.py',          # 1. Pull TEAM data
    'code/get_data/cpl_match_stats.py',         # 1. Pull MATCH data
    'code/get_data/combine_tables.py',          # 2. Combine team stats into one table and match stats into one table
    'code/pre-analysis/correlations.py',        # 3. Find weights of correlation between table stats and points
    'code/pre-analysis/zscores_strength.py'     # 4. Calculate strengths of each team based on z scores of stats compared to league and correlation weigt
]

print("--- STARTING CPL PREDICTION PIPELINE ---")

for script in scripts:
    print(f"Running: {script}...")
    # This runs the script and waits for it to finish before moving to the next
    result = subprocess.run([sys.executable, script])
    
    if result.returncode == 0:
        print(f"Successfully finished {script}\n")
    else:
        print(f"ERROR: {script} failed. Stopping pipeline.")
        break

print("--- PIPELINE COMPLETE ---")