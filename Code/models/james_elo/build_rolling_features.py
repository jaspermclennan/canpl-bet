import pandas as pd
import numpy as np
from pathlib import Path


# --- PATH SETUP ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

FEATURES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_features.csv"
BASELINE_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"

TEAM_NAME_MAP = {
    "HFX Wanderers": "Wanderers",
    "Halifax Wanderers": "Wanderers",
    "HFX Wanderers FC": "Wanderers",
    "York United": "York",
    "York United FC": "York",
    "Atlético Ottawa": "Atlético",
    "Atletico Ottawa": "Atlético",
    "Pacific": "Pacific",
    "Pacific FC": "Pacific",
    "Valour": "Valour",
    "Valour FC": "Valour",
    "Forge": "Forge",
    "Forge FC": "Forge",
    "Cavalry": "Cavalry",
    "Cavalry FC": "Cavalry",
    "Edmonton": "Edmonton",
    "FC Edmonton": "Edmonton",
}

WINDOW = 5
USE_EMA = False
EMA_ALPHA = 0.35  # used only if USE_EMA=True

def clean_team(name: str) -> str:
    return TEAM_NAME_MAP.get(str(name).strip(), str(name).strip())

def _normalize_match_key(date_val, home, away):
    """YYYY-MM-DD_Home_vs_Away (consistent with your external_factors builder fallback)."""
    d = pd.to_datetime(date_val, errors="coerce")
    if pd.isna(d):
        return None
    return f"{d.strftime('%Y-%m-%d')}_{clean_team(home)}_vs_{clean_team(away)}"

def _merge_scores(df_features: pd.DataFrame) -> pd.DataFrame:
    """Ensures HomeScore/AwayScore exist by merging from baseline."""
    if "HomeScore" in df_features.columns and "AwayScore" in df_features.columns:
        return df_features

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Cannot find baseline file at {BASELINE_FILE}")

    df_base = pd.read_csv(BASELINE_FILE)

    # Normalize baseline columns
    date_col = "Date" if "Date" in df_base.columns else ("date" if "date" in df_base.columns else None)
    if not date_col:
        raise ValueError("Baseline file missing a date column (expected Date or date).")

    # Prefer match_id if both sides have it
    if "match_id" in df_features.columns and "match_id" in df_base.columns:
        merged = df_features.merge(
            df_base[["match_id", "HomeScore", "AwayScore"]],
            on="match_id",
            how="left",
            suffixes=("", "_base"),
        )
        # If base merge worked, use it
        if merged["HomeScore"].notna().any() or merged["AwayScore"].notna().any():
            merged["HomeScore"] = merged["HomeScore"].fillna(merged.get("HomeScore_base"))
            merged["AwayScore"] = merged["AwayScore"].fillna(merged.get("AwayScore_base"))
            for c in ["HomeScore_base", "AwayScore_base"]:
                if c in merged.columns:
                    merged = merged.drop(columns=[c])
            return merged

    # Fallback: join on date + home + away
    df_features = df_features.copy()
    df_base = df_base.copy()

    df_features["_join_key"] = df_features.apply(
        lambda r: _normalize_match_key(r.get("date"), r.get("home_team"), r.get("away_team")),
        axis=1,
    )
    df_base["_join_key"] = df_base.apply(
        lambda r: _normalize_match_key(r.get(date_col), r.get("HomeTeam"), r.get("AwayTeam")),
        axis=1,
    )

    merged = df_features.merge(
        df_base[["_join_key", "HomeScore", "AwayScore"]],
        on="_join_key",
        how="left",
        suffixes=("", "_base"),
    )
    merged = merged.drop(columns=["_join_key"])
    return merged

def main():
    print("--- CALCULATING ROLLING FORM (IMPROVED) ---")

    if not FEATURES_FILE.exists():
        print(f"Missing {FEATURES_FILE}")
        return

    df_features = pd.read_csv(FEATURES_FILE)

    # Ensure date exists
    if "date" not in df_features.columns:
        raise ValueError("match_features.csv must contain a 'date' column.")

    df = _merge_scores(df_features)

    # Normalize teams
    df["home_team"] = df["home_team"].apply(clean_team)
    df["away_team"] = df["away_team"].apply(clean_team)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # team_stats: list of dicts per team
    team_hist = {}  # {team: [{"pts":..., "gd":...}, ...]}

    home_form_pts, away_form_pts, home_form_gd, away_form_gd = [], [], [], []

    def _recent_stats(team: str):
        hist = team_hist.get(team, [])
        if not hist:
            return 0.0, 0.0
        recent = hist[-WINDOW:]
        pts = sum(x["pts"] for x in recent) / len(recent)
        gd = sum(x["gd"] for x in recent) / len(recent)
        return pts, gd

    def _ema_stats(team: str):
        hist = team_hist.get(team, [])
        if not hist:
            return 0.0, 0.0
        pts_ema = 0.0
        gd_ema = 0.0
        # Iterate from oldest to newest
        for x in hist[-max(WINDOW * 3, WINDOW):]:
            pts_ema = EMA_ALPHA * x["pts"] + (1 - EMA_ALPHA) * pts_ema
            gd_ema = EMA_ALPHA * x["gd"] + (1 - EMA_ALPHA) * gd_ema
        return pts_ema, gd_ema

    print(f"   Processing {len(df)} matches...")

    for _, row in df.iterrows():
        h = clean_team(row["home_team"])
        a = clean_team(row["away_team"])

        if USE_EMA:
            h_pts, h_gd = _ema_stats(h)
            a_pts, a_gd = _ema_stats(a)
        else:
            h_pts, h_gd = _recent_stats(h)
            a_pts, a_gd = _recent_stats(a)

        home_form_pts.append(h_pts)
        home_form_gd.append(h_gd)
        away_form_pts.append(a_pts)
        away_form_gd.append(a_gd)

        # Update only if played (score exists)
        if pd.notna(row.get("HomeScore")) and pd.notna(row.get("AwayScore")):
            hs = float(row["HomeScore"])
            as_ = float(row["AwayScore"])

            if hs > as_:
                hp, ap = 3, 0
            elif hs == as_:
                hp, ap = 1, 1
            else:
                hp, ap = 0, 3

            h_gdiff = hs - as_
            a_gdiff = as_ - hs

            team_hist.setdefault(h, []).append({"pts": hp, "gd": h_gdiff})
            team_hist.setdefault(a, []).append({"pts": ap, "gd": a_gdiff})

    df["home_form_pts"] = home_form_pts
    df["away_form_pts"] = away_form_pts
    df["home_form_gd"] = home_form_gd
    df["away_form_gd"] = away_form_gd

    df["diff_form_pts"] = df["home_form_pts"] - df["away_form_pts"]
    df["diff_form_gd"] = df["home_form_gd"] - df["away_form_gd"]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved rolling features to: {OUT_FILE}")

if __name__ == "__main__":
    main()
