import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import os

# --- DYNAMIC PATHING ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

def main():
    if not DATA_PATH.exists():
        print(f"❌ Cannot find {DATA_PATH}. Run your pipeline first!")
        return

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') # Crucial: Walk forward in time

    results = []
    
    # We start after a "burn-in" period of matches to have a training base
    burn_in = 40 
    
    for i in range(burn_in, len(df)):
        # Train ONLY on matches that happened BEFORE today
        train_df = df.iloc[:i]
        current_match = df.iloc[[i]]
        
        # X is pure team diff (No Home Field Advantage included)
        X_train = train_df[['diff_total']].values
        y_train = train_df['label'].values
        
        # Train the model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)
        
        # Predict the current match
        X_test = current_match[['diff_total']].values
        probs = model.predict_proba(X_test)[0]
        
        res_map = {2: 'Home', 1: 'Draw', 0: 'Away'}
        
        results.append({
            "Season": current_match['season'].values[0],
            "Date": current_match['date'].values[0].strftime('%Y-%m-%d'),
            "Home": current_match['home_team'].values[0],
            "Away": current_match['away_team'].values[0],
            "Actual": res_map[current_match['label'].values[0]],
            "Prob_Away": round(probs[0], 3),
            "Prob_Draw": round(probs[1], 3),
            "Prob_Home": round(probs[2], 3),
            "Model_Pick": res_map[model.predict(X_test)[0]]
        })

    report = pd.DataFrame(results)
    report['Correct'] = report['Actual'] == report['Model_Pick']
    
    # Final Table Output
    print("\n" + "="*80)
    print("   NEUTRAL WALK-FORWARD EVALUATION (NO HOME FIELD ADVANTAGE)")
    print("="*80)
    print(report[['Season', 'Date', 'Home', 'Away', 'Actual', 'Prob_Home', 'Prob_Draw', 'Prob_Away', 'Model_Pick', 'Correct']].tail(20))
    
    print("\n--- ACCURACY SUMMARY ---")
    print(report.groupby('Season')['Correct'].mean().map('{:.1%}'.format))

if __name__ == "__main__":
    main()