from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_SEASONAL = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_seasonal.csv"
OUT_DIR = REPO_ROOT / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CAREER = OUT_DIR / "player_ratings_career.csv"

DECAY = 0.85  # recency per year


def main() -> None:
    if not IN_SEASONAL.exists():
        raise FileNotFoundError(f"Seasonal ratings not found: {IN_SEASONAL}")

    df = pd.read_csv(IN_SEASONAL)

    required = {"playerName", "season", "Minutes", "AttackShrunk", "DefenseShrunk", "TotalShrunk"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in seasonal ratings: {sorted(missing)}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)

    # Drop rows with NaN shrunk scores (ineligible minutes)
    df = df.dropna(subset=["season", "AttackShrunk", "DefenseShrunk", "TotalShrunk"])

    if df.empty:
        raise ValueError("No eligible rows to compute career ratings (all shrunk scores were NaN).")

    max_year = int(df["season"].max())

    df["RecencyWeight"] = (DECAY ** (max_year - df["season"])).astype(float)
    df["W"] = df["Minutes"] * df["RecencyWeight"]

    def wavg(group: pd.DataFrame, col: str) -> float:
        w = group["W"].sum()
        if w <= 0:
            return float("nan")
        return float((group[col] * group["W"]).sum() / w)

    grouped = df.groupby("playerName", as_index=False)

    out = grouped.agg(
        TotalMinutes=("Minutes", "sum"),
        SeasonsPlayed=("season", "nunique"),
    )

    out["CareerAttack"] = grouped.apply(lambda g: wavg(g, "AttackShrunk")).values
    out["CareerDefense"] = grouped.apply(lambda g: wavg(g, "DefenseShrunk")).values
    out["CareerTotal"] = grouped.apply(lambda g: wavg(g, "TotalShrunk")).values

    out = out.sort_values("CareerTotal", ascending=False)
    out.to_csv(OUT_CAREER, index=False)

    print(f"Saved career ratings: {OUT_CAREER} ({len(out)} players)")


if __name__ == "__main__":
    main()
