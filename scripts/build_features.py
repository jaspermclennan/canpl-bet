# scripts/build_features.py
import glob, json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_latest(pattern: str):
    files = sorted(glob.glob(str(RAW_DIR / pattern)))
    if not files:
        return None, None
    path = Path(files[-1])
    return path, json.loads(path.read_text())

def to_matches_df(payload: dict) -> pd.DataFrame:
    # Expecting structure from ingest_matches.py mock/real
    df = pd.json_normalize(payload.get("events", []))
    if df.empty:
        return df
    df = df.rename(columns={"dateEvent": "date"})
    df["date"] = pd.to_datetime(df["date"])
    # label: 1 if home wins, else 0 (draws count as 0 for a 2-way example)
    df["home_score"] = pd.to_numeric(df["intHomeScore"], errors="coerce").fillna(0)
    df["away_score"] = pd.to_numeric(df["intAwayScore"], errors="coerce").fillna(0)
    df["label"] = (df["home_score"] > df["away_score"]).astype(int)
    return df[["idEvent", "date", "strHomeTeam", "strAwayTeam", "label"]]

def to_odds_df(payload) -> pd.DataFrame:
    # Mock odds is a list of dicts; real API will differ — adapt when you wire it up
    df = pd.json_normalize(payload)
    if df.empty:
        return df
    return df[["idEvent", "home_moneyline", "away_moneyline"]]

def main():
    m_path, matches = load_latest("matches_*.json")
    o_path, odds = load_latest("odds_*.json")
    if matches is None or odds is None:
        raise SystemExit("Run ingest scripts first to create raw snapshots in data/raw/")

    print("Using matches:", m_path.name)
    print("Using odds:", o_path.name)

    df_m = to_matches_df(matches)
    df_o = to_odds_df(odds)

    df = df_m.merge(df_o, on="idEvent", how="left")

    # --- Starter features (replace with real features later) ---
    df["feat_bias"] = 1.0
    # Example: implied edge features later (after de-vig), elo diff, rest days, etc.

    out = df[["date", "label", "feat_bias"]].sort_values("date")
    out_path = PROC_DIR / "canpl_training.csv"
    out.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows:", len(out))

if __name__ == "__main__":
    main()
