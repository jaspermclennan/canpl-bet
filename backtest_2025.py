import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Split: Train on 2019-2024, Test on 2025
    train_df = df[df['season'] < 2025].copy()
    test_df = df[df['season'] == 2025].copy()
    
    if test_df.empty:
        print("No 2025 data found to backtest!")
        return

    # 2. Train the "Blind" Model
    X_train = train_df[['diff_total']].values
    y_train = train_df['label'].values
    model = LogisticRegression().fit(X_train, y_train)
    
    # 3. Predict 2025 using historical sensitivity (k)
    X_test = test_df[['diff_total']].values
    probs = model.predict_proba(X_test)
    
    # Identify index for Home (2), Draw (1), Away (0)
    home_idx = np.where(model.classes_ == 2)[0][0]
    draw_idx = np.where(model.classes_ == 1)[0][0]
    away_idx = np.where(model.classes_ == 0)[0][0]
    
    # 4. Create Comparison Table
    test_df['prob_home'] = probs[:, home_idx]
    # Fair Odds = 1 / Probability
    test_df['fair_odds_home'] = 1 / test_df['prob_home']
    test_df['pred_winner'] = model.predict(X_test)
    
    # Map labels back to names
    name_map = {2: 'Home', 1: 'Draw', 0: 'Away'}
    test_df['actual_result'] = test_df['label'].map(name_map)
    test_df['model_pick'] = test_df['pred_winner'].map(name_map)
    test_df['correct'] = test_df['actual_result'] == test_df['model_pick']
    
    # 5. Output Report
    cols = ['home_team', 'away_team', 'HomeScore', 'AwayScore', 'prob_home', 'fair_odds_home', 'model_pick', 'actual_result', 'correct']
    print("\n--- 2025 BLIND BACKTEST RESULTS ---")
    print(test_df[cols].head(15).to_string(index=False))
    
    accuracy = test_df['correct'].mean()
    print(f"\n Blind Accuracy for 2025: {accuracy:.1%}")

if __name__ == "__main__":
    main()