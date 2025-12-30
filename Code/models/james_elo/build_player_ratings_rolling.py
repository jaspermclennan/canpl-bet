import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

# Inputs
MATCHES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"

# Output
OUT_DIR = REPO_ROOT / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "player_ratings_rolling.csv"

# --- ELO CONFIGURATION ---
STARTING_ELO = 1500.0
K_FACTOR = 15.0  # 15.0
HOME_ADVANTAGE = 20.0 #20.0

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def main():
    print("--- 🔄 BUILDING PLAYER ELO RATINGS (WIN-BASED) ---")
    
    if not MATCHES_FILE.exists() or not LINEUPS_FILE.exists():
        print("❌ Missing input files (matches or lineups).")
        return

    # 1. Load Data
    matches = pd.read_csv(MATCHES_FILE)
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)
    
    lineups = pd.read_csv(LINEUPS_FILE)
    
    # Pre-fetch lineups
    print("   Indexing lineups...")
    lineup_map = {}
    
    for mid, group in lineups.groupby('match_id'):
        # Store both ID and Name in the map so we can retrieve names later
        lineup_map[mid] = group[['playerId', 'playerName', 'team']].to_dict('records')

    # 2. Initialize Ratings
    player_ratings = {} # { playerId: CurrentELO }
    history = []
    
    print(f"   Processing {len(matches)} matches chronologically...")

    for idx, row in matches.iterrows():
        mid = row['match_id']
        date = row['date']
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get Players involved
        players_in_match = lineup_map.get(mid, [])
        if not players_in_match:
            continue
            
        home_pids = []
        away_pids = []
        
        # We also need a way to look up the name for the history log
        pid_to_name = {}

        def clean(s): return str(s).lower().replace(" ", "")
        h_clean = clean(home_team)
        a_clean = clean(away_team)
        
        for p in players_in_match:
            pid = p['playerId']
            p_name = p['playerName']
            p_team = clean(p['team'])
            
            pid_to_name[pid] = p_name # <--- Store name mapping here
            
            if pid not in player_ratings:
                player_ratings[pid] = STARTING_ELO
            
            # Assign side
            if p_team in h_clean or h_clean in p_team:
                home_pids.append(pid)
            elif p_team in a_clean or a_clean in p_team:
                away_pids.append(pid)
        
        if not home_pids or not away_pids:
            continue

        # 3. Calculate Team ELOs
        avg_home_elo = np.mean([player_ratings[p] for p in home_pids]) + HOME_ADVANTAGE
        avg_away_elo = np.mean([player_ratings[p] for p in away_pids])
        
        # 4. Determine Outcome
        h_score = row['HomeScore']
        a_score = row['AwayScore']
        
        if h_score > a_score:
            actual_home, actual_away = 1.0, 0.0
        elif h_score == a_score:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0
            
        # 5. Calculate Expected
        exp_home = calculate_expected_score(avg_home_elo, avg_away_elo)
        exp_away = calculate_expected_score(avg_away_elo, avg_home_elo)
        
        # 6. Update Players
        # Home Update
        delta_home = K_FACTOR * (actual_home - exp_home)
        for pid in home_pids:
            history.append({
                "match_id": mid,
                "playerId": pid,
                "playerName": pid_to_name[pid], # <--- Now we save the name!
                "date": date,
                "Rating": round(player_ratings[pid], 2)
            })
            player_ratings[pid] += delta_home
            
        # Away Update
        delta_away = K_FACTOR * (actual_away - exp_away)
        for pid in away_pids:
            history.append({
                "match_id": mid,
                "playerId": pid,
                "playerName": pid_to_name[pid], # <--- Now we save the name!
                "date": date,
                "Rating": round(player_ratings[pid], 2)
            })
            player_ratings[pid] += delta_away

    # Save to CSV
    out_df = pd.DataFrame(history)
    out_df.to_csv(OUT_FILE, index=False)
    
    print(f"✅ Saved ELO ratings to: {OUT_FILE}")
    print(f"   Total Player-Match Records: {len(out_df)}")
    print("   Example (High Ratings):")
    print(out_df.sort_values('Rating', ascending=False).head(3))

if __name__ == "__main__":
    main()