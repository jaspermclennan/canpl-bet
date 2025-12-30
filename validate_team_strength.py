import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
OUT_PLOT = REPO_ROOT / "data" / "matches" / "derived" / "calibration_plot.png"

def sigmoid(x):
    # Safe sigmoid function to prevent overflow
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def main():
    if not DATA_PATH.exists(): return
    
    df = pd.read_csv(DATA_PATH)
    
    # Normalize diff_total for visualization
    # (Simple Min-Max normalization just for the plot)
    d = df['diff_total']
    df['diff_norm'] = (d - d.mean()) / d.std()

    print("\n--- SAMPLE PREDICTIONS VS ACTUAL RESULTS ---")
    # Simple logic check: Did the favorite win?
    df['predicted_winner'] = df.apply(
        lambda x: 'Home' if x['diff_total'] > 100 else ('Away' if x['diff_total'] < -100 else 'Draw'), 
        axis=1
    )
    
    label_map = {2: 'Home', 1: 'Draw', 0: 'Away'}
    df['actual_winner'] = df['label'].map(label_map)
    
    cols = ['home_team', 'away_team', 'HomeScore', 'AwayScore', 'diff_total', 'predicted_winner', 'actual_winner']
    print(df[cols].tail(10).to_string(index=False))

    # Calibration Plot (Home Win vs Field)
    plt.figure(figsize=(10, 6))
    
    # We use a simple logistic conversion for the plot line
    prob_home = sigmoid(df['diff_norm']) 
    y_true = (df['label'] == 2).astype(int)
    
    frac, mean_pred = calibration_curve(y_true, prob_home, n_bins=10)
    
    plt.plot(mean_pred, frac, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.title("Calibration Check (Normalized ELO)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.legend()
    plt.savefig(OUT_PLOT)
    print(f"\n📊 Calibration plot saved to: {OUT_PLOT}")

if __name__ == "__main__":
    main()