import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else (__file__).resolve().parent.parent.parent.parent

# calculated fatigue
EXTERNAL_FILE = REPO_ROOT / "data" / "matches" / "derived" / "external_factors.csv"
# match results
RESULTS_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
# Saved model
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("Training External Factors Model")
    
    if not EXTERNAL_FILE.exists() or not RESULTS_FILE.exists():
        print("Missing input files. Run build_fatigue_features.py first")
        return 
    
    # 1 load data
    ext_df = pd.read_csv(EXTERNAL_FILE)
    res_df = pd.read_csv(RESULTS_FILE)
    
    # 2 Merge features with results who won?
    df = pd.merge(ext_df, res_df[['match_id', 'label']], on='match_id', how='inner')
    
    # 3 Define Features (inputs to learn from need weather)
    features = [
        'fatigue_home',
        'fatigue_away', 
        'travel_km_away',
        'weather_temp',       # NEW
        'weather_rain_prob',  # NEW
        'avg_goals_home',     # NEW
        'avg_goals_away',     # NEW
        'rain_impact_home',   # NEW (The Interaction!)
        'rain_impact_away'    # NEW
    ]
    target = 'label'
    
    # filter out future games 
    train_df = df.dropna(subset[target])
    
    X = train_df[features]
    y = train_df[target]
    
    print(f"   Training on {len(X)} matches...")
    print(f"   Features: {features}")
    
    # 4. Logistic regression training
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(x, y)

    # 5. is fatigue predictive?
    probs = model.predict_proba(X)
    loss = log_loss(y, probs)
    acc = accuracy_score(y, model.predict(X))
    
    print("-" * 30)
    print(f"   Accuracy: {acc:.1%}")
    print(f"   Log Loss: {loss:.4f}")
    print("-" * 30)
    
    # 6. show impact coefficients how much each factor matters
    print("   Feature Impact (Positive = Helps Home Win, Negative = Hurts):")
    
    coeffs = model.coef_[2] # Coefficients for 'Home Win'
    for i, f in enumerate(features):
        print(f"      {f:<20}: {coeffs[i]:.4f}")

    # 7. Save
    joblib.dump(model, MODEL_DIR / "external_model.pkl")
    print(f"\n   ✅ Model saved to {MODEL_DIR / 'external_model.pkl'}")
    
    # 8. Generate Probabilities CSV 
    all_probs = model.predict_proba(ext_df[features])
    
    output_df = ext_df[['match_id']].copy()
    output_df['external_prob_home'] = all_probs[:, 2] # Probability of Home Win
    output_df['external_prob_draw'] = all_probs[:, 1]
    output_df['external_prob_away'] = all_probs[:, 0]
    
    out_file = REPO_ROOT / "data" / "matches" / "derived" / "external_predictions.csv"
    output_df.to_csv(out_file, index=False)
    print(f"Probabilities saved to {out_file}")

if __name__ == "__main__":
    main()