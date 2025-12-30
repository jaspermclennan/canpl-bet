import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
import itertools
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

MATCHES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"

def run_elo_simulation(matches, lineup_map, k_factor, home_adv):
    """Runs a fast in-memory simulation of the ELO process."""
    player_ratings = {}
    errors = []
    
    for _, row in matches.iterrows():
        mid = row['match_id']
        
        # Get Lineups
        players = lineup_map.get(mid, [])
        if not players: continue
        
        # Identify Home/Away Players
        h_clean = str(row['home_team']).lower().replace(" ", "")
        a_clean = str(row['away_team']).lower().replace(" ", "")
        
        home_ratings = []
        away_ratings = []
        
        for p in players:
            pid = p['playerId']
            p_team = str(p['team']).lower().replace(" ", "")
            rating = player_ratings.get(pid, 1500.0)
            
            if p_team in h_clean: home_ratings.append(rating)
            elif p_team in a_clean: away_ratings.append(rating)
            
        if not home_ratings or not away_ratings: continue

        # Calc Strength
        h_elo = np.mean(home_ratings) + home_adv
        a_elo = np.mean(away_ratings)
        
        # Predict Outcome (Prob Home Win)
        # 1 / (1 + 10^((Away - Home)/400))
        prob_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        
        # Track Error (Log Loss)
        # Label: 2=Home, 1=Draw, 0=Away. We simplify to Home Win (1) vs Not (0)
        actual = 1 if row['label'] == 2 else 0
        # Clip probability to avoid log(0)
        p = np.clip(prob_home, 0.01, 0.99)
        errors.append(-1 * (actual * np.log(p) + (1 - actual) * np.log(1 - p)))
        
        # Update Ratings (Using simplified Home/Draw/Away logic for speed)
        h_score, a_score = row['HomeScore'], row['AwayScore']
        if h_score > a_score: sa_h, sa_a = 1.0, 0.0
        elif h_score == a_score: sa_h, sa_a = 0.5, 0.5
        else: sa_h, sa_a = 0.0, 1.0
        
        # Expected
        ea_h = 1 / (1 + 10 ** ((a_elo - (h_elo - home_adv)) / 400)) # Raw elo for calc
        ea_a = 1 - ea_h
        
        # Batch Update
        for p in players:
            pid = p['playerId']
            p_team = str(p['team']).lower().replace(" ", "")
            rating = player_ratings.get(pid, 1500.0)
            
            if p_team in h_clean:
                player_ratings[pid] = rating + k_factor * (sa_h - ea_h)
            elif p_team in a_clean:
                player_ratings[pid] = rating + k_factor * (sa_a - ea_a)

    return np.mean(errors)

def main():
    print("--- 🧪 STARTING HYPERPARAMETER TUNING ---")
    
    # Load Data Once
    matches = pd.read_csv(MATCHES_FILE)
    matches = matches.sort_values('date').reset_index(drop=True)
    lineups = pd.read_csv(LINEUPS_FILE)
    
    # Pre-index lineups
    lineup_map = {}
    for mid, group in lineups.groupby('match_id'):
        lineup_map[mid] = group[['playerId', 'team']].to_dict('records')

    # Grid Search
    # We will test all combinations of these values
    k_values = [10, 15, 20, 25, 30, 40]
    hfa_values = [20, 35, 50, 65, 80]
    
    best_score = float('inf')
    best_params = {}
    
    print(f"Testing {len(k_values) * len(hfa_values)} combinations...")
    
    for k, hfa in itertools.product(k_values, hfa_values):
        loss = run_elo_simulation(matches, lineup_map, k, hfa)
        print(f"K={k:<2} | HFA={hfa:<2} | Loss={loss:.4f}")
        
        if loss < best_score:
            best_score = loss
            best_params = {'k': k, 'hfa': hfa}
            
    print("-" * 40)
    print(f"🏆 BEST PARAMETERS FOUND:")
    print(f"   K_FACTOR: {best_params['k']}")
    print(f"   HOME_ADVANTAGE: {best_params['hfa']}")
    print(f"   Lowest Log Loss: {best_score:.4f}")
    print("-" * 40)
    print("👉 Update your build_player_ratings_rolling.py with these values!")

if __name__ == "__main__":
    main()