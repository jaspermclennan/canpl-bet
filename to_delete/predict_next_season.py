import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

# Inputs
ROLLING_FILE = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_rolling.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"
MODEL_DIR = REPO_ROOT / "models"

# Output
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "2026_season_predictions.csv"

def get_latest_rosters():
    """Determines who is on which team based on late-2025 data."""
    print("   Building 2026 Rosters from 2025 data...")
    
    # 1. Get latest rating for every player
    ratings = pd.read_csv(ROLLING_FILE)
    ratings['date'] = pd.to_datetime(ratings['date'])
    # Sort by date and take the last rating for each player
    latest_ratings = ratings.sort_values('date').groupby('playerId').tail(1)
    latest_ratings = latest_ratings[['playerId', 'playerName', 'Rating']]
    
    # 2. Get latest team for every player
    lineups = pd.read_csv(LINEUPS_FILE)
    lineups['date'] = pd.to_datetime(lineups['date'])
    # We only care about games from the last 3 months of 2025 (Projected Roster)
    recent_games = lineups[lineups['date'] > '2025-08-01']
    
    if recent_games.empty:
        # Fallback if no recent data (use all data)
        recent_games = lineups
        
    latest_teams = recent_games.sort_values('date').groupby('playerId').tail(1)
    latest_teams = latest_teams[['playerId', 'team']]
    
    # 3. Merge
    roster_df = latest_teams.merge(latest_ratings, on='playerId')
    return roster_df

def calculate_team_strength(roster_df):
    """Calculates the 'Best XI' strength for each team."""
    team_strengths = {}
    
    print("\n--- 2026 POWER RANKINGS (Start of Season) ---")
    print(f"{'Rank':<5} | {'Team':<20} | {'ELO Strength (Top 11)':<10}")
    print("-" * 55)
    
    # We assume the top 11 players play full minutes
    for team, group in roster_df.groupby('team'):
        # Get top 14 players (Starter + Subs rotation proxy)
        top_players = group.sort_values('Rating', ascending=False).head(14)
        
        # Weighted Strength: Top 11 get 100% weight, next 3 get 30% (Depth)
        # This is a proxy for the 'sum(Rating * w)' logic in match_team_strength.py
        starters = top_players.iloc[:11]['Rating'].sum()
        subs = top_players.iloc[11:]['Rating'].sum() * 0.3
        
        total_strength = starters + subs
        team_strengths[team] = total_strength

    # Sort and Print
    sorted_teams = sorted(team_strengths.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (team, strength) in enumerate(sorted_teams, 1):
        print(f"{rank:<5} | {team:<20} | {strength:.1f}")
        
    return team_strengths

def predict_matchups(team_strengths):
    """Generates odds for a hypothetical Round Robin."""
    
    # Load Model
    model_path = MODEL_DIR / "logistic_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        print("\n❌ Model files not found. Run build_probability_model.py first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("\n--- 2026 OPENING MATCHUP PROJECTIONS ---")
    print(f"{'Home':<15} vs {'Away':<15} | {'Home Win':<8} | {'Draw':<8} | {'Away Win':<8} | {'Fair Odds (H/D/A)'}")
    print("-" * 95)
    
    teams = list(team_strengths.keys())
    
    # Generate every possible matchup (Home vs Away)
    predictions = []
    
    for home in teams:
        for away in teams:
            if home == away: continue
            
            # 1. Calculate Diff (Pure Talent)
            # We assume Form Diff is 0.0 at start of season (Everyone starts equal)
            diff_total = team_strengths[home] - team_strengths[away]
            diff_form_pts = 0.0
            diff_form_gd = 0.0
            
            # 2. Prepare Feature Vector
            # Order must match training: ['diff_total', 'diff_form_pts', 'diff_form_gd']
            # Note: If your training used different features, adjust this list!
            features = np.array([[diff_total]]) 
            
            # Check feature count compatibility
            # In our last fix, we trained on 1 feature (diff_total) in build_probability_model.py
            # If you updated that to use form, we need form here. 
            # Assuming build_probability_model.py used ONLY diff_total for the saved model:
            
            try:
                features_scaled = scaler.transform(features)
                probs = model.predict_proba(features_scaled)[0] # [Away, Draw, Home]
                
                # Odds = 1 / Probability
                odds_h = 1 / probs[2]
                odds_d = 1 / probs[1]
                odds_a = 1 / probs[0]
                
                print(f"{home:<15} vs {away:<15} | {probs[2]:.1%}     | {probs[1]:.1%}     | {probs[0]:.1%}     | {odds_h:.2f} / {odds_d:.2f} / {odds_a:.2f}")
                
                predictions.append({
                    'Home': home, 'Away': away,
                    'Prob_Home': probs[2], 'Prob_Draw': probs[1], 'Prob_Away': probs[0],
                    'Odds_Home': odds_h, 'Odds_Draw': odds_d, 'Odds_Away': odds_a
                })
                
            except ValueError:
                print("⚠️ Feature mismatch. Ensure predict_next_season.py matches the features in build_probability_model.py")
                return

    # Save
    pd.DataFrame(predictions).to_csv(OUT_FILE, index=False)
    print(f"\n✅ Predictions saved to {OUT_FILE}")

def main():
    if not ROLLING_FILE.exists():
        print("Missing rolling ratings file.")
        return

    roster = get_latest_rosters()
    strengths = calculate_team_strength(roster)
    predict_matchups(strengths)

if __name__ == "__main__":
    main()