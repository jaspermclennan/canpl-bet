import pandas as pd
import sys
import joblib
import numpy as np

# Load the model once at the top level when the module is imported
# This makes subsequent function calls much faster.
MODEL_PATH = 'data/analysis/cpl_ml_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

def calculate_ml_probs(home_val, away_val):
    """
    Core engine for ML Prediction.
    Returns probabilities as [home_p, draw_p, away_p] (0.0 to 1.0)
    """
    if model is None:
        raise FileNotFoundError(f"ML Model not found at {MODEL_PATH}")

    # Calculate the gap
    gap = home_val - away_val
    
    # ML expects a DataFrame with the correct feature name
    # Result order: 0=Home Win, 1=Draw, 2=Away Win
    gap_df = pd.DataFrame([[gap]], columns=['Prior_Gap'])
    probs = model.predict_proba(gap_df)[0]
    
    return probs

# --- THIS PART RUNS ONLY IF YOU EXECUTE THIS FILE DIRECTLY ---
if __name__ == "__main__":
    team_ids = {1: 'Cavalry', 2: 'Forge', 3: 'Atlético Ottawa', 4: 'HFX Wanderers',
                5: 'Inter Toronto', 6: 'Pacific', 7: 'Vancouver FC', 8: 'FC Supra du Québec'}

    if len(sys.argv) < 4:
        print("Usage: python3 ml_odds.py <YEAR> <HOME_ID> <AWAY_ID>")
        sys.exit(1)

    TARGET_YEAR = sys.argv[1]
    HOME_TEAM = team_ids[int(sys.argv[2])]
    AWAY_TEAM = team_ids[int(sys.argv[3])]

    # Load strengths file
    df_teams = pd.read_csv(f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv')
    
    # Get values
    h_val = df_teams[df_teams["Team"] == HOME_TEAM][f"Historical Prior for {TARGET_YEAR} Season"].values[0]
    a_val = df_teams[df_teams["Team"] == AWAY_TEAM][f"Historical Prior for {TARGET_YEAR} Season"].values[0]

    # Run Prediction
    probs = calculate_ml_probs(h_val, a_val)

    print(f"--- ML Prediction: {HOME_TEAM} vs {AWAY_TEAM} ---")
    print(f"Home: {probs[0]*100:.2f}% | Draw: {probs[1]*100:.2f}% | Away: {probs[2]*100:.2f}%")