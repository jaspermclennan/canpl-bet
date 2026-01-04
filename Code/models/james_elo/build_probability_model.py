import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import joblib
from pathlib import Path
import os

# --- FIX PATHS HERE ---
# Since this file is in Code/models/james_elo/, we need 4 parents to get to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# This must match the OUT_FILE from build_targets.py
DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("--- 🤖 TRAINING PROBABILITY MODEL (SCALED) ---")
    
    if not DATA_FILE.exists():
        print(f"❌ Critical Error: Input file not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Validation Check
    if 'label' not in df.columns:
        print(f"❌ Error: 'label' column missing. Columns found: {list(df.columns)}")
        print("   Did build_targets.py run successfully?")
        return

    # Filter out future matches (where we don't have a result yet)
    # If label is missing (NaN), drop it
    train_df = df.dropna(subset=['label'])
    
    # Feature Selection (Must match what we trained on)
    features = ['diff_total']
    target = 'label'
    
    X = train_df[features]
    y = train_df[target]
    
    # Scaling (Important for some models, good practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression (The "Calibrator")
    # We use this to turn "Diff +100" into "65% Probability"
    clf = LogisticRegression(solver='lbfgs', C=1.0)
    clf.fit(X_scaled, y)
    
    # Evaluate
    preds = clf.predict_proba(X_scaled)
    loss = log_loss(y, preds)
    acc = accuracy_score(y, clf.predict(X_scaled))
    
    print(f"   Matches used: {len(train_df)}")
    print(f"   Log Loss: {loss:.4f}")
    print(f"   Accuracy: {acc:.1%}")
    print(f"   Coefficients: {clf.coef_}")
    
    # Save Artifacts
    joblib.dump(clf, MODEL_DIR / "logistic_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"✅ Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()