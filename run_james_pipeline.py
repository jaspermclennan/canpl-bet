import subprocess
from pathlib import Path

# Define the root of your project
REPO_ROOT = Path(__file__).resolve().parent

# Define the main 'models' folder
MODELS_DIR = REPO_ROOT / "Code" / "models"

# The strictly ordered sequence for 2026 Season Readiness
PIPELINE = [
    # 1. Player Ratings 
    MODELS_DIR / "james_elo" / "build_player_ratings_rolling.py",
    
    # 2. Match Setup (Shared logic)
    MODELS_DIR / "build_assumed_lineups.py",
    MODELS_DIR / "build_match_team_strength.py",
    
    # 3. Feature Engineering
    MODELS_DIR / "build_match_features.py", 
    MODELS_DIR / "james_elo" / "build_rolling_features.py", 

    # 4. Model Training
    MODELS_DIR / "build_targets.py",          
    MODELS_DIR / "james_elo" / "build_probability_model.py", 
    
    # 5. Validation 
    REPO_ROOT / "validate_team_strength.py"    
]

def main():
    print("STARTING DATA PIPELINE")
    for script in PIPELINE:
        if script.exists():
            print(f"Running {script.name}...")
            subprocess.run(["python3", str(script)], check=True)
        else:
            print(f"File not found: {script}")
            print(f"Expected at: {script}") 
    print("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()