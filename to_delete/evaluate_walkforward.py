import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # <--- NEW
from sklearn.calibration import calibration_curve
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

# Input: The file created by build_rolling_features.py
DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"
PLOT_DIR = REPO_ROOT / "data" / "matches" / "derived"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "walkforward_calibration.png"

def main():
    if not DATA_PATH.exists():
        print(f"❌ Missing file: {DATA_PATH}")
        print("   Please run 'build_rolling_features.py' first to generate the dataset.")
        return

    print("--- 🔄 STARTING WALK-FORWARD EVALUATION (SCALED + FORM) ---")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    results = []
    START_INDEX = 50
    total_matches = len(df)
    
    print(f"   Processing {total_matches - START_INDEX} matches...")

    for i in range(START_INDEX, total_matches):
        train_df = df.iloc[:i]
        current_match = df.iloc[[i]].copy()
        
        # Features: Talent Gap + Rolling Form Gaps
        features = ['diff_total', 'diff_form_pts', 'diff_form_gd']
        
        X_train = train_df[features].values
        y_train = train_df['label'].values
        
        X_test = current_match[features].values

        # --- THE FIX: SCALING ---
        # We must scale the data so 'Goal Diff' doesn't overpower 'Team Strength'
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- THE FIX: MAX ITERATIONS ---
        # Increased max_iter to 1000 to stop ConvergenceWarnings
        model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        probs = model.predict_proba(X_test_scaled)[0]
        pred_label = model.predict(X_test_scaled)[0]
        
        label_map = {2: 'Home', 1: 'Draw', 0: 'Away'}
        actual_gd = current_match['HomeScore'].values[0] - current_match['AwayScore'].values[0]
        
        results.append({
            "Season": current_match['season'].values[0],
            "Date": current_match['date'].dt.strftime('%Y-%m-%d').values[0],
            "Home": current_match['home_team'].values[0],
            "Away": current_match['away_team'].values[0],
            "Actual_GD": actual_gd,
            "Actual": label_map[current_match['label'].values[0]],
            "Pick": label_map[pred_label],
            "Prob_Home": round(probs[2], 3),
            "Prob_Draw": round(probs[1], 3),
            "Prob_Away": round(probs[0], 3),
            "Correct": pred_label == current_match['label'].values[0],
            "Label_Numeric": current_match['label'].values[0]
        })

    # --- REPORTING ---
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("   WALK-FORWARD PREDICTION LOG (SAMPLE)")
    print("="*80)
    cols = ['Date', 'Home', 'Away', 'Actual_GD', 'Actual', 'Prob_Home', 'Prob_Draw', 'Prob_Away', 'Pick', 'Correct']
    print(res_df[cols].tail(15).to_string(index=False))
    
    print("\n--- ACCURACY BY SEASON ---")
    print(res_df.groupby('Season')['Correct'].mean().map('{:.1%}'.format))
    print(f"\n   Overall Accuracy: {res_df['Correct'].mean():.1%}")

    # Plotting
    plt.figure(figsize=(10, 6))
    y_true_home = (res_df['Label_Numeric'] == 2).astype(int)
    prob_home = res_df['Prob_Home']
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true_home, prob_home, n_bins=8)
    
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Your Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability of Home Win")
    plt.ylabel("Actual Frequency of Home Win")
    plt.title("Model Reality Check (Scaled Data)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"\n📊 Calibration Plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()