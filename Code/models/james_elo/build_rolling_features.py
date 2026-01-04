import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- PATH SETUP ---
# We are deep in Code/models/james_elo/, so 4 parents needed
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# INPUT 1: The file with ELO gaps (from build_match_features.py)
FEATURES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_features.csv"

# INPUT 2: The Master file (to get Scores back if missing)
BASELINE_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"

# OUTPUT: The file with Form added
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"

TEAM_NAME_MAP = {
    "HFX Wanderers": "Wanderers",
    "Halifax Wanderers": "Wanderers",
    "HFX Wanderers FC": "Wanderers",
    "York United": "York",
    "York United FC": "York",
    "Atlético Ottawa": "Atlético",
    "Atletico Ottawa": "Atlético",
    "Pacific": "Pacific",
    "Pacific FC": "Pacific",
    "Valour": "Valour",
    "Valour FC": "Valour",
    "Forge": "Forge",
    "Forge FC": "Forge",
    "Cavalry": "Cavalry",
    "Cavalry FC": "Cavalry",
    "Edmonton": "Edmonton",
    "FC Edmonton": "Edmonton"
}

def clean_team(name):
    return TEAM_NAME_MAP.get(str(name).strip(), str(name).strip())

def main():
    print("--- 🔄 CALCULATING ROLLING FORM ---")
    
    if not FEATURES_FILE.exists():
        print(f"❌ Missing {FEATURES_FILE}")
        return

    # 1. Load Features (ELO gaps)
    df_features = pd.read_csv(FEATURES_FILE)
    
    # 2. Load Master (Scores) and Merge if necessary
    # The error happened because 'HomeScore' was missing. We fix that here.
    if 'HomeScore' not in df_features.columns:
        print("   ⚠️ Scores missing in input. Merging from master file...")
        if not BASELINE_FILE.exists():
            print(f"❌ Critical: Cannot find master file at {BASELINE_FILE}")
            return
            
        df_base = pd.read_csv(BASELINE_FILE)
        
        # Normalize keys for join (Date + HomeTeam)
        df_features['join_date'] = pd.to_datetime(df_features['date']).dt.strftime('%Y-%m-%d')
        df_features['join_h'] = df_features['home_team'].apply(clean_team)
        
        # Check column names in base file
        date_col = 'Date' if 'Date' in df_base.columns else 'date'
        df_base['join_date'] = pd.to_datetime(df_base[date_col]).dt.strftime('%Y-%m-%d')
        df_base['join_h'] = df_base['HomeTeam'].apply(clean_team)
        
        # Merge only the scores
        df_merged = pd.merge(
            df_features,
            df_base[['join_date', 'join_h', 'HomeScore', 'AwayScore']],
            on=['join_date', 'join_h'],
            how='left'
        )
        
        # Cleanup join columns
        df_merged = df_merged.drop(columns=['join_date', 'join_h'])
    else:
        df_merged = df_features.copy()

    # 3. Calculate Form
    # We must sort by date to calculate "Past 5 Games" correctly
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_merged = df_merged.sort_values('date')
    
    # Dictionary to track team history
    team_stats = {} # {team_name: [{'pts': 3, 'gd': 2}, ...]}

    # Lists to store the new features
    home_form_pts = []
    away_form_pts = []
    home_form_gd = []
    away_form_gd = []
    
    print(f"   Processing {len(df_merged)} matches...")

    # Iterate row by row
    for idx, row in df_merged.iterrows():
        h_team = clean_team(row['home_team'])
        a_team = clean_team(row['away_team'])
        
        # A. RETRIEVE Past Form (Average of Last 5 games)
        def get_recent_stats(team):
            history = team_stats.get(team, [])
            if len(history) == 0:
                return 0.0, 0.0 # No history = 0 form
            
            recent = history[-5:] # Last 5
            pts = sum(x['pts'] for x in recent) / len(recent) # Avg Points
            gd = sum(x['gd'] for x in recent) / len(recent)   # Avg Goal Diff
            return pts, gd

        h_pts, h_gd = get_recent_stats(h_team)
        a_pts, a_gd = get_recent_stats(a_team)
        
        home_form_pts.append(h_pts)
        home_form_gd.append(h_gd)
        away_form_pts.append(a_pts)
        away_form_gd.append(a_gd)
        
        # B. UPDATE History (for the NEXT time this team plays)
        # We only update if the game has actually been played (has a score)
        if pd.notna(row['HomeScore']):
            h_s = row['HomeScore']
            a_s = row['AwayScore']
            
            # Points (3 for Win, 1 for Draw, 0 for Loss)
            if h_s > a_s: hp, ap = 3, 0
            elif h_s == a_s: hp, ap = 1, 1
            else: hp, ap = 0, 3
            
            # Goal Difference
            h_gdiff = h_s - a_s
            a_gdiff = a_s - h_s
            
            if h_team not in team_stats: team_stats[h_team] = []
            if a_team not in team_stats: team_stats[a_team] = []
            
            team_stats[h_team].append({'pts': hp, 'gd': h_gdiff})
            team_stats[a_team].append({'pts': ap, 'gd': a_gdiff})
            
    # 4. Add Columns to DataFrame
    df_merged['home_form_pts'] = home_form_pts
    df_merged['away_form_pts'] = away_form_pts
    df_merged['home_form_gd'] = home_form_gd
    df_merged['away_form_gd'] = away_form_gd
    
    # Calculate Diffs (Home - Away)
    df_merged['diff_form_pts'] = df_merged['home_form_pts'] - df_merged['away_form_pts']
    df_merged['diff_form_gd'] = df_merged['home_form_gd'] - df_merged['away_form_gd']
    
    # 5. Save
    df_merged.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved rolling features to: {OUT_FILE}")
    print(f"   (Added 'diff_form_pts' and 'diff_form_gd')")

if __name__ == "__main__":
    main()