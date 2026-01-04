import pandas as pd
import joblib
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_FILE = REPO_ROOT / "models" / "xgboost_model.pkl"
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "james_ml_predictions.csv"

def main():
    if not MODEL_FILE.exists():
        print("Train the model first")
        return

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    
    features = ['diff_total', 'diff_form_pts', 'diff_form_gd']
    
    # Predict
    probs = model.predict_proba(df[features])
    
    # Format Output
    output = df[['match_id']].copy()
    output['prob_away'] = probs[:, 0]
    output['prob_draw'] = probs[:, 1]
    output['prob_home'] = probs[:, 2]
    
    output.to_csv(OUT_FILE, index=False)
    print(f"ML Predictions saved to {OUT_FILE}")

if __name__ == "__main__":
    main()