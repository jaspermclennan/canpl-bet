from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = (
    cwd if cwd.name == "canpl-bet" else Path(__file__).resolve().parent.parent.parent.parent
)

EXTERNAL_FILE = REPO_ROOT / "data" / "matches" / "derived" / "external_factors.csv"
RESULTS_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

MODEL_OUT = REPO_ROOT / "data" / "matches" / "derived" / "external_model.joblib"
PREDS_OUT = REPO_ROOT / "data" / "matches" / "derived" / "external_predictions.csv"


# --- NORMALIZATION HELPERS ---
_TEAM_MAP = {
    # Common long -> short keys used across your pipeline
    "hfx wanderers": "wanderers",
    "halifax wanderers": "wanderers",
    "york united": "york",
    "atletico ottawa": "atletico",
    "atlético ottawa": "atletico",
    "forge fc": "forge",
    "cavalry fc": "cavalry",
    "pacific fc": "pacific",
    "valour fc": "valour",
    "vancouver fc": "vancouver",
    "fc edmonton": "edmonton",
    "edmonton fc": "edmonton",
}


def _strip_accents(s: str) -> str:
    """Convert unicode accents to closest ASCII (Atlético -> Atletico)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_team(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = re.sub(r"\s+", " ", s)
    s = _strip_accents(s).lower()

    # remove common suffix/prefix noise
    s = re.sub(r"\bfc\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()

    # map to canonical short name where possible
    return _TEAM_MAP.get(s, s)


def _parse_match_id(mid: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to parse:
      YYYY-MM-DD_Home_vs_Away
      YYYY-MM-DD Home vs Away
      YYYY-MM-DD_Home v Away (rare)
    """
    if mid is None or (isinstance(mid, float) and np.isnan(mid)):
        return None
    s = str(mid).strip()
    if not s:
        return None

    # Normalize separators lightly
    s2 = s.replace(" VS ", " vs ").replace(" Vs ", " vs ").replace(" v ", " vs ")
    m = re.match(r"(\d{4}-\d{2}-\d{2})[_\s]+(.+?)(?:_vs_| vs )(.+)$", s2)
    if not m:
        return None
    date_str, home, away = m.group(1), m.group(2), m.group(3)
    return date_str, home, away


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_best_by_keywords(df: pd.DataFrame, include: list[str], exclude: Optional[list[str]] = None) -> Optional[str]:
    """
    Heuristic column picker: choose the column with the highest score based on keyword inclusion.
    """
    exclude = exclude or []
    cols = list(df.columns)

    best = None
    best_score = -1
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in exclude):
            continue
        score = 0
        for kw in include:
            if kw in cl:
                score += 1
        if score > best_score:
            best_score = score
            best = c

    return best if best_score > 0 else None


def _build_match_key_from_parts(date_str: str, home: str, away: str) -> str:
    return f"{date_str}_{_norm_team(home)}_vs_{_norm_team(away)}"


def _ensure_match_key_ext(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Create match_key + date for external_factors.csv (usually only has match_id)."""
    ext_df = ext_df.copy()

    if "match_id" not in ext_df.columns:
        raise ValueError("external_factors.csv must contain 'match_id'.")

    keys, dates = [], []
    for mid in ext_df["match_id"]:
        parsed = _parse_match_id(mid)
        if parsed:
            d, h, a = parsed
            keys.append(_build_match_key_from_parts(d, h, a))
            dates.append(d)
        else:
            keys.append("")
            dates.append("")

    ext_df["match_key"] = keys
    ext_df["date"] = pd.to_datetime(dates, errors="coerce")
    return ext_df


def _ensure_match_key_res(res_df: pd.DataFrame) -> pd.DataFrame:
    """Create match_key + date for match_model_ready.csv using multiple fallbacks."""
    res_df = res_df.copy()

    # 1) Attempt parse from match_id if present
    if "match_id" in res_df.columns:
        keys, dates = [], []
        failures = 0
        for mid in res_df["match_id"]:
            parsed = _parse_match_id(mid)
            if parsed:
                d, h, a = parsed
                keys.append(_build_match_key_from_parts(d, h, a))
                dates.append(d)
            else:
                failures += 1
                keys.append("")
                dates.append("")
        res_df["match_key"] = keys

        # If most rows failed to parse, fall back to date/home/away columns
        if failures > 0.8 * len(res_df):
            res_df = res_df.drop(columns=["match_key"])
        else:
            if "date" not in res_df.columns:
                res_df["date"] = pd.to_datetime(dates, errors="coerce")
            else:
                res_df["date"] = pd.to_datetime(res_df["date"], errors="coerce")
            return res_df

    # 2) Build from date + home + away columns (explicit names first)
    date_col = _pick_first(res_df, ["date", "match_date", "game_date"])
    home_col = _pick_first(res_df, ["home_team", "home", "homeTeam", "team_home", "home_name"])
    away_col = _pick_first(res_df, ["away_team", "away", "awayTeam", "team_away", "away_name"])

    # 3) Heuristic fallback if explicit names are missing
    if not date_col:
        date_col = _pick_best_by_keywords(res_df, include=["date"], exclude=["update", "created"])
    if not home_col:
        # prefer "home" + "team" columns, avoid "home_prob"/"home_odds"
        home_col = _pick_best_by_keywords(res_df, include=["home", "team"], exclude=["prob", "odds", "elo", "score", "goals", "xg"])
        if not home_col:
            home_col = _pick_best_by_keywords(res_df, include=["home"], exclude=["prob", "odds", "elo", "score", "goals", "xg"])
    if not away_col:
        away_col = _pick_best_by_keywords(res_df, include=["away", "team"], exclude=["prob", "odds", "elo", "score", "goals", "xg"])
        if not away_col:
            away_col = _pick_best_by_keywords(res_df, include=["away"], exclude=["prob", "odds", "elo", "score", "goals", "xg"])

    if not (date_col and home_col and away_col):
        # Print columns to make debugging immediate
        raise ValueError(
            "Could not build match_key in match_model_ready.csv.\n"
            f"Detected columns: {list(res_df.columns)}\n"
            "Need either:\n"
            "  - a parseable 'match_id' like 'YYYY-MM-DD_Team_vs_Team', OR\n"
            "  - columns for (date + home team + away team).\n"
            "If your columns use different names, update the candidate lists in this script."
        )

    res_df[date_col] = pd.to_datetime(res_df[date_col], errors="coerce")

    # Build raw key from parts then normalize by parsing again
    raw = (
        res_df[date_col].dt.strftime("%Y-%m-%d").fillna("")
        + "_"
        + res_df[home_col].astype(str)
        + "_vs_"
        + res_df[away_col].astype(str)
    )

    norm_keys = []
    for mk in raw.astype(str):
        parsed = _parse_match_id(mk)
        if parsed:
            d, h, a = parsed
            norm_keys.append(_build_match_key_from_parts(d, h, a))
        else:
            norm_keys.append("")

    res_df["match_key"] = norm_keys
    res_df["date"] = res_df[date_col]
    return res_df


def main():
    print("--- TRAINING EXTERNAL FACTORS MODEL ---")

    if not EXTERNAL_FILE.exists():
        raise FileNotFoundError(f"Missing {EXTERNAL_FILE}. Run build_fatigue_features.py first.")
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing {RESULTS_FILE}.")

    ext_df = pd.read_csv(EXTERNAL_FILE)
    res_df = pd.read_csv(RESULTS_FILE)

    # Label column detection (default: 'label')
    label_col = _pick_first(res_df, ["label", "result_label", "outcome_label", "y"])
    if not label_col:
        raise ValueError(
            "match_model_ready.csv must contain a label column (expected one of: "
            "'label', 'result_label', 'outcome_label', 'y')."
        )

    # Build robust join key
    ext_df = _ensure_match_key_ext(ext_df)
    res_df = _ensure_match_key_res(res_df)

    # If match_key building failed on results, show immediately
    if "match_key" not in res_df.columns:
        raise ValueError("Internal error: match_key not present after processing match_model_ready.csv.")

    # Merge labels onto external factors
    df = ext_df.merge(res_df[["match_key", label_col, "date"]], on="match_key", how="inner")
    df = df.rename(columns={label_col: "label"})

    print(f"Merged rows (external_factors ∩ match_model_ready): {len(df)}")
    if len(df) == 0:
        print("No rows merged. Debug info:")
        print("External sample keys:", ext_df["match_key"].dropna().head(8).tolist())
        print("Results sample keys:", res_df["match_key"].dropna().head(8).tolist())
        print("Results columns:", list(res_df.columns))
        raise ValueError(
            "No rows matched between external_factors.csv and match_model_ready.csv.\n"
            "Most common causes:\n"
            "  1) match_model_ready.csv lacks usable (date/home/away) or match_id fields\n"
            "  2) team naming differs (e.g., 'HFX Wanderers' vs 'Wanderers') beyond the current TEAM_MAP\n"
            "  3) match_id format differs from YYYY-MM-DD_Home_vs_Away\n"
            "Fix: confirm which columns hold date/home/away in match_model_ready.csv and align naming."
        )

    # Feature set (use what's present)
    candidate_features = [
        "fatigue_home",
        "fatigue_away",
        "travel_km_away",
        "tz_change_away",
        "weather_temp",
        "weather_rain_prob",
        "avg_goals_home",
        "avg_goals_away",
        "rain_impact_home",
        "rain_impact_away",
    ]
    features = [c for c in candidate_features if c in df.columns]
    if not features:
        raise ValueError("No usable features found in merged dataframe.")

    # Ensure numeric
    for c in features + ["label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing label/features
    df = df.dropna(subset=features + ["label"]).copy()
    print(f"Rows after dropna(features+label): {len(df)}")
    if len(df) < 50:
        print("Warning: very small training set after cleaning. Check upstream joins/features.")

    # Time-based split if date is available
    # Note: merge may create date_x/date_y if both inputs have a date column.
    date_col_merged = None
    for _cand in ["date", "date_y", "date_x"]:
        if _cand in df.columns:
            date_col_merged = _cand
            break
    if date_col_merged is None:
        # Fallback: parse date from match_key prefix "YYYY-MM-DD_"
        df["date"] = pd.to_datetime(df["match_key"].astype(str).str.slice(0, 10), errors="coerce")
    else:
        df["date"] = pd.to_datetime(df[date_col_merged], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if len(train_df) == 0:
        raise ValueError("No training rows after split.")

    X_train = train_df[features].values
    y_train = train_df["label"].astype(int).values
    X_test = test_df[features].values
    y_test = test_df["label"].astype(int).values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(multi_class="multinomial", max_iter=5000, C=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test) if len(test_df) else float("nan")
    print(f"Test accuracy: {acc:.3f}" if not np.isnan(acc) else "No test split to score.")

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")

    # Generate predictions for all merged rows
    probs = model.predict_proba(df[features].values)
    classes = list(model.named_steps["clf"].classes_)

    out = df[["match_id", "date"]].copy()
    if set(classes) == {0, 1, 2}:
        idx0, idx1, idx2 = classes.index(0), classes.index(1), classes.index(2)
        out["p_away"] = probs[:, idx0]
        out["p_draw"] = probs[:, idx1]
        out["p_home"] = probs[:, idx2]
    else:
        for j, c in enumerate(classes):
            out[f"p_class_{c}"] = probs[:, j]

    PREDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDS_OUT, index=False)
    print(f"Saved predictions to: {PREDS_OUT}")
    print(out.tail())


if __name__ == "__main__":
    main()
