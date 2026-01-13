import pandas as pd
import sys
from scipy.stats import poisson
import numpy as np

def calculate_poisson_probs(home_val, away_val, actual_avg_goals):
    """
    Core math engine for Poisson. 
    Returns probabilities as [home_p, draw_p, away_p] (0.0 to 1.0)
    """
    # 1. Get the gap
    gap = (home_val - away_val) / 50

    # 2. Give each team their 'Target Goals' (Lambda)
    lambda_h = actual_avg_goals + (gap / 2)
    lambda_a = actual_avg_goals - (gap / 2)

    # 3. Probability distribution for 0-8 goals
    home_chances = [poisson.pmf(i, lambda_h) for i in range(9)]
    away_chances = [poisson.pmf(i, lambda_a) for i in range(9)]

    # 4. Create the grid
    matrix = np.outer(home_chances, away_chances)

    # 5. Sum the 'buckets' and normalize to sum to 1.0
    home_p = np.sum(np.tril(matrix, -1)) 
    draw_p = np.sum(np.diag(matrix))      
    away_p = np.sum(np.triu(matrix, 1))  
    
    total = home_p + draw_p + away_p
    return np.array([home_p/total, draw_p/total, away_p/total])

# --- THIS PART RUNS ONLY IF YOU EXECUTE THIS FILE DIRECTLY ---
if __name__ == "__main__":
    team_ids = {1: 'Cavalry', 2: 'Forge', 3: 'Atlético Ottawa', 4: 'HFX Wanderers',
                5: 'Inter Toronto', 6: 'Pacific', 7: 'Vancouver FC', 8: 'FC Supra du Québec'}

    # Load matches for the environment
    df_matches = pd.read_csv('data/matches/combined/matches_combined.csv')
    df_matches.columns = df_matches.columns.str.strip().str.title()
    total_goals = df_matches['Homescore'].sum() + df_matches['Awayscore'].sum()
    actual_avg_goals = total_goals / (len(df_matches) * 2)

    TARGET_YEAR = sys.argv[1]
    HOME_TEAM = team_ids[int(sys.argv[2])]
    AWAY_TEAM = team_ids[int(sys.argv[3])]

    df_teams = pd.read_csv(f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv')
    h_val = df_teams[df_teams["Team"] == HOME_TEAM][f"Historical Prior for {TARGET_YEAR} Season"].values[0]
    a_val = df_teams[df_teams["Team"] == AWAY_TEAM][f"Historical Prior for {TARGET_YEAR} Season"].values[0]

    probs = calculate_poisson_probs(h_val, a_val, actual_avg_goals)

    print(f"--- Poisson Results: {HOME_TEAM} vs {AWAY_TEAM} ---")
    print(f"Win: {probs[0]*100:.2f}% | Draw: {probs[1]*100:.2f}% | Win: {probs[2]*100:.2f}%")