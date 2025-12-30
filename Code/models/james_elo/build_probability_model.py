import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib  # To save the model/scaler for later use
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        print(f"Missing {DATA_PATH}")
        return

    print("--- Training Probability Model (Scaled) ---")
    df = pd.read_csv(DATA_PATH)
    
    # We want to predict Home Win (Label=2) vs Not Home Win
    # (Simplified for the 'k' factor display, though actual model is multinomial)
    X = df[['diff_total']].values
    y = df['label'].values

    # 1. Scale the Inputs (Crucial for ELO data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Train Logistic Regression
    clf = LogisticRegression(solver='lbfgs', C=1.0)
    clf.fit(X_scaled, y)
    
    # Save for later
    joblib.dump(clf, MODEL_DIR / "logistic_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # 3. Interpret Results
    # We simulate a "Standard" strong team (+1 Std Dev) to see the impact
    test_gap = [[1000.0]] # E.g., a 1000 point ELO gap
    test_gap_scaled = scaler.transform(test_gap)
    probs = clf.predict_proba(test_gap_scaled)[0]
    
    print("-" * 30)
    print(f"Model Trained on {len(df)} matches.")
    print(f"Classes: {clf.classes_} (0=Away, 1=Draw, 2=Home)")
    print("-" * 30)
    print("Example Prediction for +1000 ELO Advantage:")
    print(f"Home Win: {probs[2]:.1%}")
    print(f"Draw:     {probs[1]:.1%}")
    print(f"Away Win: {probs[0]:.1%}")
    print("-" * 30)

if __name__ == "__main__":
    main()