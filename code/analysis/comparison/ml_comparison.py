import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

# 1. SETUP
TARGET_YEAR = 2025
STRENGTHS_PATH = f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv'
MODEL_PATH = 'data/analysis/cpl_ml_model.pkl'
MATCHES_PATH = 'data/matches/combined/matches_combined.csv'

# 2. LOAD DATA & MODEL
df_matches = pd.read_csv(MATCHES_PATH)
df_teams = pd.read_csv(STRENGTHS_PATH)
model = joblib.load(MODEL_PATH)

# Filter for the 2025 season
df_2025 = df_matches[df_matches['Season'] == TARGET_YEAR].copy()

# 3. DEFINE EVALUATION LOGIC
def get_actual_result(row):
    """Encodes outcome: 0=Home Win, 1=Draw, 2=Away Win"""
    if row['Homescore'] > row['Awayscore']:
        return 0
    elif row['Homescore'] == row['Awayscore']:
        return 1
    else:
        return 2

# Pre-calculate a dictionary for quick strength lookups
strength_col = f"Historical Prior for {TARGET_YEAR} Season"
team_strengths = dict(zip(df_teams['Team'], df_teams[strength_col]))

results = []

print(f"--- Evaluating 2025 Season ({len(df_2025)} matches) ---")

for idx, row in df_2025.iterrows():
    home_team = row['Hometeam']
    away_team = row['Awayteam']
    
    # Get strengths and calculate gap
    try:
        home_val = team_strengths[home_team]
        away_val = team_strengths[away_team]
        gap = home_val - away_val
        
        # ML Prediction
        gap_df = pd.DataFrame([[gap]], columns=['Prior_Gap'])
        probs = model.predict_proba(gap_df)[0] # [Home_P, Draw_P, Away_P]
        
        # Actual result
        actual = get_actual_result(row)
        
        results.append({
            'Home': home_team,
            'Away': away_team,
            'Prob_H': probs[0],
            'Prob_D': probs[1],
            'Prob_A': probs[2],
            'Predicted_Class': np.argmax(probs),
            'Actual_Class': actual
        })
    except KeyError as e:
        print(f"Skipping match: Team {e} not found in strengths file.")

# 4. COMPUTE METRICS
eval_df = pd.DataFrame(results)

# Accuracy (How often was the highest probability outcome correct?)
acc = accuracy_score(eval_df['Actual_Class'], eval_df['Predicted_Class'])

# Log Loss (How accurate were the probabilities?)
# We pass the full 3-column probability array
ll = log_loss(eval_df['Actual_Class'], eval_df[['Prob_H', 'Prob_D', 'Prob_A']].values, labels=[0,1,2])

# 5. PRINT SUMMARY
print("-" * 40)
print(f"OVERALL ACCURACY: {acc:.2%}")
print(f"LOG LOSS:         {ll:.4f}")
print("-" * 40)

# breakdown by outcome
cm = confusion_matrix(eval_df['Actual_Class'], eval_df['Predicted_Class'])
print("Confusion Matrix (Rows=Actual, Cols=Predicted):")
print("      H   D   A")
print(f"H: {cm[0]}")
print(f"D: {cm[1]}")
print(f"A: {cm[2]}")