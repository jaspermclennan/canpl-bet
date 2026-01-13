import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import joblib
from pathlib import Path
import os
import json


# --- PATH SETUP ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Recommended feature set (the script will auto-intersect with available columns)
RECOMMENDED_FEATURES = [
    # core elo signal
    "diff_total",
    # form
    "diff_form_pts",
    "diff_form_gd",
    # external factors (if merged in later)
    "fatigue_home",
    "fatigue_away",
    "travel_km_home",
    "travel_km_away",
    "tz_change_home",
    "tz_change_away",
    "weather_temp",
    "weather_rain_prob",
    "rain_impact_home",
    "rain_impact_away",
]

LABEL_COL_CANDIDATES = ["label", "result_label", "outcome_label", "y"]
DATE_COL_CANDIDATES = ["date", "Date", "match_date"]

def _pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _time_split(df: pd.DataFrame, date_col: str, test_frac: float = 0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < 50:
        # avoid ultra-small test sets
        test_frac = 0.25
    cut = int(np.floor((1.0 - test_frac) * n))
    cut = max(1, min(cut, n - 1))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test, df[date_col].iloc[cut]

def main():
    print("--- TRAINING PROBABILITY MODEL (IMPROVED) ---")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input file not found at: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Label
    label_col = _pick_first_existing(df.columns, LABEL_COL_CANDIDATES)
    if not label_col:
        raise ValueError(f"No label column found. Tried: {LABEL_COL_CANDIDATES}. Found: {list(df.columns)}")

    df = df.dropna(subset=[label_col]).copy()

    # Date (for time split)
    date_col = _pick_first_existing(df.columns, DATE_COL_CANDIDATES)
    if not date_col:
        raise ValueError(f"No date column found. Tried: {DATE_COL_CANDIDATES}. Found: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Feature selection
    features = [c for c in RECOMMENDED_FEATURES if c in df.columns]
    if len(features) == 0:
        raise ValueError(
            "No usable features found. "
            f"Recommended: {RECOMMENDED_FEATURES}. Found: {list(df.columns)}"
        )

    # Drop rows with missing feature values
    df_model = df.dropna(subset=features + [label_col]).copy()
    if df_model.empty:
        raise ValueError("No training rows after dropping NaNs (features+label).")

    # Train/test split by time
    train_df, test_df, cutoff_ts = _time_split(df_model, date_col, test_frac=0.2)

    X_train = train_df[features].astype(float).values
    y_train = train_df[label_col].astype(int).values
    X_test = test_df[features].astype(float).values
    y_test = test_df[label_col].astype(int).values

    # Model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            max_iter=1000,
        ))
    ])

    pipe.fit(X_train, y_train)

    # Metrics
    train_probs = pipe.predict_proba(X_train)
    test_probs = pipe.predict_proba(X_test)

    train_loss = log_loss(y_train, train_probs, labels=pipe.named_steps["clf"].classes_)
    test_loss = log_loss(y_test, test_probs, labels=pipe.named_steps["clf"].classes_)

    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))

    classes = pipe.named_steps["clf"].classes_.tolist()

    print(f"   Rows used: {len(df_model)}")
    print(f"   Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"   Cutoff date (test starts): {cutoff_ts}")
    print(f"   Features ({len(features)}): {features}")
    print(f"   Classes: {classes}")
    print(f"   Train LogLoss: {train_loss:.4f} | Train Acc: {train_acc:.1%}")
    print(f"   Test  LogLoss: {test_loss:.4f} | Test  Acc: {test_acc:.1%}")

    artifact = {
        "pipeline": pipe,
        "features": features,
        "classes": classes,
        "date_col": date_col,
        "label_col": label_col,
        "cutoff_date": str(cutoff_ts),
        "metrics": {
            "train_log_loss": float(train_loss),
            "test_log_loss": float(test_loss),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "n_total": int(len(df_model)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }
    }

    out_path = MODEL_DIR / "probability_model_artifact.pkl"
    joblib.dump(artifact, out_path)

    # Backwards-compatible: also save the separate pieces your other scripts expect
    joblib.dump(pipe.named_steps["clf"], MODEL_DIR / "logistic_model.pkl")
    joblib.dump(pipe.named_steps["scaler"], MODEL_DIR / "scaler.pkl")

    meta_path = MODEL_DIR / "probability_model_meta.json"
    meta_path.write_text(json.dumps({k: artifact[k] for k in ["features","classes","date_col","label_col","cutoff_date","metrics"]}, indent=2))

    print(f"Saved model artifact to: {out_path}")
    print(f"Saved meta to: {meta_path}")
    print(f"Saved logistic_model.pkl + scaler.pkl to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
