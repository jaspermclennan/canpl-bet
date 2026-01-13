import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import log_loss

# 1. IMPORT YOUR "EXPERT" FUNCTIONS
from pre_match_odds_poisson import calculate_poisson_probs
from pre_match_odds_ml import calculate_ml_probs

# 2. CONFIGURATION
team_ids = {
    1: 'Cavalry', 2: 'Forge', 3: 'Atlético Ottawa', 4: 'HFX Wanderers',
    5: 'Inter Toronto', 6: 'Pacific', 7: 'Vancouver FC', 8: 'FC Supra du Québec'
}

def find_optimal_weights(validation_year, matches_df, avg_goals):
    """
    Simulates the validation year to find the best Poisson/ML balance.
    Prints individual and ensemble Log Loss.
    """
    val_strengths_path = f'data/analysis/predict_{validation_year}_from_historic.csv'
    
    if not os.path.exists(val_strengths_path):
        print(f"--- Warning: {validation_year} data not found. Using default 0.6/0.4 split. ---")
        return 0.60

    print(f"--- Optimizing Weights using {validation_year} Data ---")
    
    df_val_teams = pd.read_csv(val_strengths_path)
    val_strength_col = f"Historical Prior for {validation_year} Season"
    val_strengths = dict(zip(df_val_teams['Team'], df_val_teams[val_strength_col]))
    
    df_val_matches = matches_df[matches_df['Season'] == validation_year].copy()
    
    poi_probs, ml_probs, actual_results = [], [], []

    for _, row in df_val_matches.iterrows():
        try:
            h_v, a_v = val_strengths[row['Hometeam']], val_strengths[row['Awayteam']]
            poi_probs.append(calculate_poisson_probs(h_v, a_v, avg_goals))
            ml_probs.append(calculate_ml_probs(h_v, a_v))
            actual = 0 if row['Homescore'] > row['Awayscore'] else (1 if row['Homescore'] == row['Awayscore'] else 2)
            actual_results.append(actual)
        except KeyError: continue

    P, M, Y = np.array(poi_probs), np.array(ml_probs), np.array(actual_results)
    
    # Calculate individual baseline losses
    loss_poi = log_loss(Y, P, labels=[0, 1, 2])
    loss_ml = log_loss(Y, M, labels=[0, 1, 2])
    
    best_score, best_w = float('inf'), 0.5
    shrinkage_factor = 0.05 

    for i in range(101):
        w = i / 100.0
        ensemble_p = (P * w) + (M * (1.0 - w))
        raw_loss = log_loss(Y, ensemble_p, labels=[0, 1, 2])
        penalty = shrinkage_factor * abs(w - 0.5)
        total_score = raw_loss + penalty
        
        if total_score < best_score:
            best_score, best_w, final_raw_loss = total_score, w, raw_loss
            
    print(f"Poisson Loss: {loss_poi:.4f} | ML Loss: {loss_ml:.4f}")
    print(f"Optimal Balance: {best_w:.2f} Poi / {1.0-best_w:.2f} ML (Ensemble Loss: {final_raw_loss:.4f})")
    return best_w

def run_prediction(target_year, home_id, away_id):
    strengths_path = f'data/analysis/predict_{target_year}_from_historic.csv'
    matches_path = 'data/matches/combined/matches_combined.csv'

    try:
        df_teams = pd.read_csv(strengths_path)
        df_matches = pd.read_csv(matches_path)
    except Exception as e:
        print(f"Data Error: {e}"); return

    df_matches.columns = df_matches.columns.str.strip().str.title()
    total_goals = df_matches['Homescore'].sum() + df_matches['Awayscore'].sum()
    actual_avg_goals = total_goals / (len(df_matches) * 2)

    val_year = int(target_year) - 1
    w_poi = find_optimal_weights(val_year, df_matches, actual_avg_goals)
    w_ml = 1.0 - w_poi

    # 2. RUN TARGET PREDICTION (Pure Strength, No HFA)
    strength_col = f"Historical Prior for {target_year} Season"
    team_strengths = dict(zip(df_teams['Team'], df_teams[strength_col]))
    
    home_team, away_team = team_ids[home_id], team_ids[away_id]
    h_v, a_v = team_strengths[home_team], team_strengths[away_team]

    poi_p = calculate_poisson_probs(h_v, a_v, actual_avg_goals)
    ml_p  = calculate_ml_probs(h_v, a_v)
    final_p = (poi_p * w_poi) + (ml_p * w_ml)

    print(f"\n" + "="*45)
    print(f" {target_year} CONSENSUS: {home_team} vs {away_team}")
    print("="*45)
    print(f"{'Win:':<20} {final_p[0]:.2%}")
    print(f"{'Draw:':<20} {final_p[1]:.2%}")
    print(f"{'Away Win:':<20} {final_p[2]:.2%}")
    print("="*45)
    print(f" FAIR ODDS: H: {1/final_p[0]:.2f} | D: {1/final_p[1]:.2f} | A: {1/final_p[2]:.2f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 ensemble.py <YEAR> <HOME_ID> <AWAY_ID>")
    else:
        run_prediction(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))