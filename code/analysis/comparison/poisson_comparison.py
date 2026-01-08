import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

# 1. SETUP & DATA LOADING
TARGET_YEAR = 2025
MATCHES_PATH = 'data/matches/combined/matches_combined.csv'
STRENGTHS_PATH = f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv'

df_matches = pd.read_csv(MATCHES_PATH)
df_matches.columns = df_matches.columns.str.strip().str.title()
df_teams = pd.read_csv(STRENGTHS_PATH)

# Filter for 2025 season
df_2025 = df_matches[df_matches['Season'] == TARGET_YEAR].copy()

# 2. CALC LEAGUE AVG (Required for Poisson Lambda)
total_goals = df_matches['Homescore'].sum() + df_matches['Awayscore'].sum()
actual_avg_goals = total_goals / (len(df_matches) * 2)

# Pre-calculate team strengths dictionary for speed
strength_col = f"Historical Prior for {TARGET_YEAR} Season"
team_strengths = dict(zip(df_teams['Team'], df_teams[strength_col]))

def get_actual_result(row):
    if row['Homescore'] > row['Awayscore']: return 0 # Home
    if row['Homescore'] == row['Awayscore']: return 1 # Draw
    return 2 # Away

results = []

print(f"--- Evaluating Poisson Model: 2025 Season ---")

for idx, row in df_2025.iterrows():
    home_team = row['Hometeam']
    away_team = row['Awayteam']
    
    try:
        # Get strengths
        h_val = team_strengths[home_team]
        a_val = team_strengths[away_team]
        
        # --- POISSON LOGIC ---
        gap = (h_val - a_val) / 50
        lambda_h = actual_avg_goals + (gap / 2)
        lambda_a = actual_avg_goals - (gap / 2)
        
        # Calculate goal probabilities (up to 8 goals)
        h_chances = [poisson.pmf(i, lambda_h) for i in range(9)]
        a_chances = [poisson.pmf(i, lambda_a) for i in range(9)]
        
        # Create score matrix
        matrix = np.outer(h_chances, a_chances)
        
        # Calculate Win/Draw/Loss Probabilities
        prob_h = np.sum(np.tril(matrix, -1))
        prob_d = np.sum(np.diag(matrix))
        prob_a = np.sum(np.triu(matrix, 1))
        
        actual = get_actual_result(row)
        
        results.append({
            'Prob_H': prob_h,
            'Prob_D': prob_d,
            'Prob_A': prob_a,
            'Predicted_Class': np.argmax([prob_h, prob_d, prob_a]),
            'Actual_Class': actual
        })
    except KeyError:
        continue

# 3. COMPUTE METRICS
eval_df = pd.DataFrame(results)
acc = accuracy_score(eval_df['Actual_Class'], eval_df['Predicted_Class'])
ll = log_loss(eval_df['Actual_Class'], eval_df[['Prob_H', 'Prob_D', 'Prob_A']].values, labels=[0,1,2])

print("-" * 40)
print(f"POISSON ACCURACY: {acc:.2%}")
print(f"POISSON LOG LOSS: {ll:.4f}")
print("-" * 40)

cm = confusion_matrix(eval_df['Actual_Class'], eval_df['Predicted_Class'])
print("Confusion Matrix:")
print("      H   D   A")
print(f"H: {cm[0]}")
print(f"D: {cm[1]}")
print(f"A: {cm[2]}")