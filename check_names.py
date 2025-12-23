import pandas as pd
from pathlib import Path
import os

print("\n[DEBUG] Script is starting...")

current_dir = Path(os.getcwd())
player_file = current_dir / "data" / "players" / "cleaned" / "cpl_players_all_seasons_cleaned.csv"

print(f"[DEBUG] Looking for file at: {player_file}")

def audit():
    if not player_file.exists():
        print(f"ERROR: File does not exist at {player_file}")
        return

    print("[DEBUG] File found! Reading CSV...")
    df = pd.read_csv(player_file)
    
    print("\n" + "="*30)
    print("   PLAYER BASE DATA AUDIT")
    print("="*30)
    
    # Check seasons
    if 'season' not in df.columns:
        print(f"ERROR: 'season' column not found. Columns are: {df.columns.tolist()}")
        return

    seasons = sorted(df['season'].unique())
    print(f"Seasons available: {[int(s) for s in seasons]}")
    
    # Check 2022
    df_2022 = df[df['season'] == 2022].copy()
    
    if df_2022.empty:
        print("ERROR: No data found for the 2022 season!")
    else:
        if 'team' not in df_2022.columns:
            print(f"ERROR: 'team' column not found. Columns are: {df_2022.columns.tolist()}")
            return
            
        df_2022['team'] = df_2022['team'].fillna("MISSING_NAME").astype(str)
        teams_2022 = sorted(df_2022['team'].unique().tolist())
        print(f"Teams found in 2022: {teams_2022}")
        
        missing_count = (df_2022['team'] == "MISSING_NAME").sum()
        if missing_count > 0:
            print(f"WARNING: {missing_count} player rows have NO team name in 2022!")
            
    print("="*30 + "\n")

audit()