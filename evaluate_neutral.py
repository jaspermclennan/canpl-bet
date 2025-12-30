import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

def main():
    if not DATA_PATH.exists():
        print("❌ Run your pipeline first to create match_model_ready.csv")
        return

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    results = []
    # Start after 40 matches to have a training baseline
    for i in range(40, len(df)):
        train_df = df.iloc[:i]
        match = df.iloc[[i]]
        
        # Pure Team Differential (No HFA added)
        X_train = train_df[['diff_total']].values
        y_train = train_df['label'].values
        
        # Train Model on Past Data
        model = LogisticRegression(multi_class='multinomial').fit(X_train, y_train)
        
        # Predict current match
        X_test = match[['diff_total']].values
        probs = model.predict_proba(X_test)[0]
        
        res_map = {2: 'Home', 1: 'Draw', 0: 'Away'}
        h_score = int(match['HomeScore'].values[0])
        a_score = int(match['AwayScore'].values[0])
        
        results.append({
            "Season": match['season'].values[0],
            "Home": match['home_team'].values[0],
            "Away": match['away_team'].values[0],
            "Actual_GD": h_score - a_score, # <--- GOAL DIFFERENTIAL
            "Result": res_map[match['label'].values[0]],
            "Prob_Away": round(probs[np.where(model.classes_ == 0)[0][0]], 3),
            "Prob_Draw": round(probs[np.where(model.classes_ == 1)[0][0]], 3),
            "Prob_Home": round(probs[np.where(model.classes_ == 2)[0][0]], 3),
            "Model_Pick": res_map[model.predict(X_test)[0]]
        })

    report = pd.DataFrame(results)
    report['Correct'] = report['Result'] == report['Model_Pick']
    
    print("\n--- NEUTRAL WALK-FORWARD EVALUATION (2022-2025) ---")
    cols = ['Season', 'Home', 'Away', 'Actual_GD', 'Result', 'Prob_Home', 'Prob_Draw', 'Prob_Away', 'Model_Pick']
    print(report[cols].tail(25).to_string(index=False))
    
    print("\n--- NEUTRAL ACCURACY BY SEASON ---")
    print(report.groupby('Season')['Correct'].mean().map('{:.1%}'.format))

if __name__ == "__main__": main()