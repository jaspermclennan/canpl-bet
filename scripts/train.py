import argparse
from pathlib import Path
import pandas as pd
from sportsbook_ml.features import add_basic_features
from sportsbook_ml.models import fit_calibrated_logit
from sportsbook_ml.eval import evaluate_probs
from sportsbook_ml.utils import chronological_split

def main(args):
    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df.sort_values("date")
    df = add_basic_features(df)

    train, test = chronological_split(df, frac=0.8)
    X_train = train.drop(columns=["label", "date"])
    y_train = train["label"].astype(int)

    X_test = test.drop(columns=["label", "date"])
    y_test = test["label"].astype(int)

    model = fit_calibrated_logit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_probs(y_test, p)
    print("Metrics:", metrics)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    pd.DataFrame({
        "date": test["date"].to_numpy(),
        "pred": p,
        "y": y_test.to_numpy()
    }).to_csv(out / "oof.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to processed CSV with columns: date,label,features...")
    main(parser.parse_args())
