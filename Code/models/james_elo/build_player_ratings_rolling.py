import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- PATH SETUP ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# INPUT 1: The Master Match File
MATCHES_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"

# INPUT 2: The Lineups
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"

# OUTPUT: Where we save the ratings
OUT_FILE = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_rolling.csv"

# SETTINGS
K_FACTOR = 20.0
HOME_ADVANTAGE = 50.0

# --- TEAM MAPPING (Must match your other files) ---
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

def clean_team_name(name):
    """Standardizes team names to ensure IDs match."""
    clean = str(name).strip()
    return TEAM_NAME_MAP.get(clean, clean)

def main():
    print("--- 🔄 BUILDING PLAYER ELO RATINGS (WIN-BASED) ---")
    
    # 1. Load Data
    if not MATCHES_FILE.exists():
        print(f"❌ Critical Error: Match file not found at {MATCHES_FILE}")
        return
    
    matches = pd.read_csv(MATCHES_FILE)
    
    # 2. Fix Column Names (Score & Date)
    if 'HomeScore' not in matches.columns and 'home_score' in matches.columns:
        matches = matches.rename(columns={'home_score': 'HomeScore', 'away_score': 'AwayScore'})
    
    # Ensure Date is datetime
    date_col = 'Date' if 'Date' in matches.columns else 'date'
    matches['date'] = pd.to_datetime(matches[date_col])
    matches = matches.sort_values('date').reset_index(drop=True)

    # 3. GENERATE MATCH_ID (The Missing Link)
    # We construct it exactly like: YYYY_YYYY-MM-DD_Home_vs_Away
    print("   Generating match IDs...")
    matches['season_id'] = matches['date'].dt.year.astype(str)
    matches['date_id'] = matches['date'].dt.strftime('%Y-%m-%d')
    matches['h_clean'] = matches['HomeTeam'].apply(clean_team_name)
    matches['a_clean'] = matches['AwayTeam'].apply(clean_team_name)
    
    matches['match_id'] = (
        matches['season_id'] + "_" + 
        matches['date_id'] + "_" + 
        matches['h_clean'] + "_vs_" + 
        matches['a_clean']
    )

    # 4. Load & Index Lineups
    lineups = pd.read_csv(LINEUPS_FILE)
    print("   Indexing lineups...")
    lineup_map = {}
    for mid, group in lineups.groupby('match_id'):
        lineup_map[mid] = group[['playerId', 'team']].to_dict('records')

    # 5. Initialize Ratings
    player_ratings = {} # {playerId: 1500.0}
    history_records = []

    print(f"   Processing {len(matches)} matches chronologically...")

    # 6. The ELO Loop
    for _, row in matches.iterrows():
        mid = row['match_id']
        
        # Check if we have a lineup for this generated ID
        players = lineup_map.get(mid, [])
        if not players:
            # Debugging: Uncomment if you suspect mismatches
            # print(f"⚠️ Warning: No lineup found for {mid}")
            continue
            
        # Identify Teams (Clean version for matching)
        h_clean = row['h_clean'].lower().replace(" ", "")
        a_clean = row['a_clean'].lower().replace(" ", "")
        
        h_ratings = []
        a_ratings = []
        active_ids_home = []
        active_ids_away = []
        
        for p in players:
            pid = p['playerId']
            p_team = str(p['team']).strip().replace("FC", "").strip()
            p_team_clean = TEAM_NAME_MAP.get(p_team, p_team).lower().replace(" ", "")

            rating = player_ratings.get(pid, 1500.0)
            
            # Save Pre-Match Rating
            history_records.append({
                'match_id': mid,
                'playerId': pid,
                'team': p['team'],
                'date': row['date'],
                'Rating': rating
            })
            
            # Assign to Home or Away bucket
            # We use loose matching "in" to handle slight naming diffs
            if p_team_clean in h_clean or h_clean in p_team_clean:
                h_ratings.append(rating)
                active_ids_home.append(pid)
            elif p_team_clean in a_clean or a_clean in p_team_clean:
                a_ratings.append(rating)
                active_ids_away.append(pid)
        
        if not h_ratings or not a_ratings:
            continue

        # Calculate ELO Deltas
        h_elo = np.mean(h_ratings) + HOME_ADVANTAGE
        a_elo = np.mean(a_ratings)
        
        ea_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        ea_a = 1 - ea_h
        
        h_score = row['HomeScore']
        a_score = row['AwayScore']
        
        if h_score > a_score:
            sa_h, sa_a = 1.0, 0.0
        elif h_score == a_score:
            sa_h, sa_a = 0.5, 0.5
        else:
            sa_h, sa_a = 0.0, 1.0
            
        # Update Home Players
        for pid in active_ids_home:
            current_rating = player_ratings.get(pid, 1500.0)
            player_ratings[pid] = current_rating + K_FACTOR * (sa_h - ea_h)
            
        # Update Away Players
        for pid in active_ids_away:
            current_rating = player_ratings.get(pid, 1500.0)
            player_ratings[pid] = current_rating + K_FACTOR * (sa_a - ea_a)

    # 7. Save
    out_df = pd.DataFrame(history_records)
    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    print(f"✅ Saved ELO ratings to: {OUT_FILE}")
    print(f"   Total Player-Match Records: {len(out_df)}")

if __name__ == "__main__":
    main()