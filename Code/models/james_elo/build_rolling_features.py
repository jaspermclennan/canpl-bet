import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

# Input: The file with all matches and results
IN_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
# Output: The same file, but enriched with "Form" columns
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"

def get_last_n_games(history, n=5):
    """Calculates avg points and avg GD from the last N games."""
    if len(history) == 0:
        return 0.0, 0.0
    
    recent = history[-n:]
    avg_pts = sum(g['pts'] for g in recent) / len(recent)
    avg_gd = sum(g['gd'] for g in recent) / len(recent)
    return avg_pts, avg_gd

def main():
    if not IN_FILE.exists():
        print(f"❌ Missing {IN_FILE}")
        return

    df = pd.read_csv(IN_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Dictionary to store team history
    # Format: "Pacific": [{ 'pts': 3, 'gd': 1 }, { 'pts': 0, 'gd': -2 } ...]
    team_history = {}

    # New columns
    home_form_pts = []
    home_form_gd = []
    away_form_pts = []
    away_form_gd = []

    print(f"--- Calculating Rolling Form for {len(df)} matches ---")

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Initialize if new
        if home not in team_history: team_history[home] = []
        if away not in team_history: team_history[away] = []

        # 1. GET FORM (Before this match played)
        h_pts, h_gd = get_last_n_games(team_history[home], n=5)
        a_pts, a_gd = get_last_n_games(team_history[away], n=5)
        
        home_form_pts.append(h_pts)
        home_form_gd.append(h_gd)
        away_form_pts.append(a_pts)
        away_form_gd.append(a_gd)

        # 2. UPDATE HISTORY (After match played)
        h_score = row['HomeScore']
        a_score = row['AwayScore']
        
        # Calculate points
        if h_score > a_score:
            h_p, a_p = 3, 0
        elif h_score == a_score:
            h_p, a_p = 1, 1
        else:
            h_p, a_p = 0, 3
            
        team_history[home].append({'pts': h_p, 'gd': h_score - a_score})
        team_history[away].append({'pts': a_p, 'gd': a_score - h_score})

    # Attach columns
    df['home_form_pts'] = home_form_pts
    df['home_form_gd'] = home_form_gd
    df['away_form_pts'] = away_form_pts
    df['away_form_gd'] = away_form_gd

    # Calculate Form Differentials
    df['diff_form_pts'] = df['home_form_pts'] - df['away_form_pts']
    df['diff_form_gd'] = df['home_form_gd'] - df['away_form_gd']

    df.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved rolling features to: {OUT_FILE}")
    print("   (Added 'diff_form_pts' and 'diff_form_gd')")

if __name__ == "__main__":
    main()