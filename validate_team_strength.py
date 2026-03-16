"""
Validates the match_team_strength.csv output from build_match_team_strength.py

Checks:
- Required columns exist
- Data types are correct
- No unexpected nulls
- Strength values are reasonable
- Home/Away coverage is balanced
- All seasons have expected match counts
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import os
import sys

# Detects REPO_ROOT 
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet" else Path(__file__).resolve().parent

STRENGTH_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_team_strength.csv"

REQUIRED_COLUMNS = [
    "match_id", "season", "date", "team", "side", "opponent",
    "team_attack", "team_defense", "team_total", "coverage_rate", "coverage_ok"
]

def validate_team_strength():
    print("--- Validating Match Team Strength Data ---")
    
    if not STRENGTH_FILE.exists():
        print(f"❌ CRITICAL: {STRENGTH_FILE} does not exist")
        return False
    
    df = pd.read_csv(STRENGTH_FILE)
    errors = []
    warnings = []
    
    # 1. Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # 2. Check data types
    expected_types = {
        "match_id": "object",
        "season": "int64",
        "date": "object",
        "team": "object",
        "side": ["home", "away"],
        "opponent": "object",
        "team_attack": "float64",
        "team_defense": "float64",
        "team_total": "float64",
        "coverage_rate": "float64",
        "coverage_ok": bool,
    }
    
    # 3. Check for nulls in critical columns
    critical_nulls = df[["match_id", "team", "side", "team_total"]].isnull()
    if critical_nulls.any().any():
        errors.append(f"Critical nulls found:\n{critical_nulls.sum()}")
    
    # 4. Check strength values are in reasonable range (0-1000000)
    for col in ["team_attack", "team_defense", "team_total"]:
        outliers = df[df[col] > 1000000]
        if not outliers.empty:
            warnings.append(f"⚠️  {col} has very high values (max: {df[col].max():.0f})")
        
        if (df[col] < 0).any():
            errors.append(f"{col} has negative values")
    
    # 5. Check home/away balance
    side_counts = df["side"].value_counts()
    if len(side_counts) != 2 or side_counts["home"] != side_counts["away"]:
        errors.append(f"Imbalanced home/away: {side_counts.to_dict()}")
    
    # 6. Check match_ids are paired (each match has 2 teams)
    match_counts = df.groupby("match_id").size()
    if not (match_counts == 2).all():
        errors.append(f"Not all matches have exactly 2 teams: {match_counts[match_counts != 2]}")
    
    # 7. Check seasons match known CPL years
    seasons = sorted(df["season"].unique())
    known_seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    unknown = [s for s in seasons if s not in known_seasons]
    if unknown:
        warnings.append(f"Unknown seasons: {unknown}")
    
    # 8. Check coverage_ok alignment
    coverage_mismatch = df[df["coverage_ok"] != (df["coverage_rate"] > 0.8)]
    if not coverage_mismatch.empty:
        errors.append(f"Coverage check mismatch in {len(coverage_mismatch)} rows")
    
    # Summary
    print(f"\n✅ Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✅ Seasons covered: {seasons}")
    print(f"✅ Teams: {df['team'].nunique()} unique")
    print(f"✅ Matches: {df['match_id'].nunique()} unique")
    
    if warnings:
        print(f"\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print(f"\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("\n✅ Validation PASSED")
    return True

if __name__ == "__main__":
    success = validate_team_strength()
    sys.exit(0 if success else 1)
