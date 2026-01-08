import subprocess
import sys

# --- THE FIX: MANDATORY YEAR INPUT ---
# Check if the user provided an argument (sys.argv[0] is the script name, sys.argv[1] is the year)
if len(sys.argv) < 2:
    print("\n[!] ERROR: You must provide a target year to run the pipeline.")
    print("Usage: python3 code/pipeline/pipeline.py <YEAR>")
    sys.exit(1)  # Stop the script entirely

TARGET_YEAR = sys.argv[1]

scripts = [
    'code/get_data/cpl_team_stats.py',
    'code/get_data/cpl_match_stats.py',
    'code/get_data/combine_tables.py',
    'code/pre-analysis/correlations.py',
    'code/pre-analysis/zscores_strength.py',
    'code/pre-analysis/predicted_strengths.py'
]

print(f"--- STARTING CPL PREDICTION PIPELINE FOR {TARGET_YEAR} ---")

for script in scripts:
    print(f"Running: {script}...")
    # Passes the mandatory TARGET_YEAR to every script
    result = subprocess.run([sys.executable, script, TARGET_YEAR])
    
    if result.returncode != 0:
        print(f"ERROR: {script} failed. Stopping pipeline.")
        sys.exit(1)

print("--- PIPELINE COMPLETE ---")