# @AUTHOUR: Jasper McLennan
import pandas as pd
import numpy as np
from scipy.stats import poisson
import sys

# 1. TEAM MAPPING (1-8)
team_map = {
    1: 'Atlético Ottawa',
    2: 'Forge',
    3: 'Cavalry',
    4: 'HFX Wanderers',
    5: 'Inter Toronto',
    6: 'Pacific',
    7: 'Vancouver FC',
    8: 'FC Supra du Québec'
}

# 2. HANDLE COMMAND LINE INPUT (Format: 1 2 2026)
if len(sys.argv) < 4:
    print("Usage: python predict.py [HomeID] [AwayID] [Year]")
    print("Example: python predict.py 1 8 2026")
    sys.exit(1)

h_id = int(sys.argv[1])
a_id = int(sys.argv[2])
target_year = int(sys.argv[3])

home_team = team_map[h_id]
away_team = team_map[a_id]

# 3. PATHS & DATA
STRENGTHS_PATH = f'data/analysis/prediction_sets/team_power_rankings/predict_{target_year}_from_historic.csv'
MATCHES_PATH = 'data/raw/match/combined/matches_combined.csv'

try:
    df_teams = pd.read_csv(STRENGTHS_PATH)
    df_matches = pd.read_csv(MATCHES_PATH)
    df_matches.columns = df_matches.columns.str.strip().str.title()
except FileNotFoundError:
    print("Error: Required CSV files not found.")
    sys.exit(1)

# 4. MATH ENGINE (THE TING)
past_matches = df_matches[df_matches['Season'] < target_year]
avg_goals = (past_matches['Homescore'].sum() + past_matches['Awayscore'].sum()) / (len(past_matches) * 2)

strength_col = f"Historical Prior for {target_year} Season"
h_val = df_teams.loc[df_teams['Team'] == home_team, strength_col].values[0]
a_val = df_teams.loc[df_teams['Team'] == away_team, strength_col].values[0]

# Standard /50 scaling
gap = (h_val - a_val) / 50
lambda_h = avg_goals + (gap / 2)
lambda_a = avg_goals - (gap / 2)

# Poisson Matrix
h_probs = [poisson.pmf(i, lambda_h) for i in range(9)]
a_probs = [poisson.pmf(i, lambda_a) for i in range(9)]
matrix = np.outer(h_probs, a_probs)

# 5. FINAL CALCULATION & OUTPUT
prob_h = np.sum(np.tril(matrix, -1))
prob_d = np.sum(np.diag(matrix))
prob_a = np.sum(np.triu(matrix, 1))

print(f"\n{home_team} vs {away_team} ({target_year})")
print("-" * 35)
print(f"HOME WIN: {prob_h:.1%}")
print(f"DRAW:     {prob_d:.1%}")
print(f"AWAY WIN: {prob_a:.1%}")
print("-" * 35)