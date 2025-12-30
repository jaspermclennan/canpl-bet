from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PLAYER_BASE = REPO_ROOT / "data" / "players" / "cleaned" / "cpl_players_all_seasons_cleaned.csv"
MATCHES_RAW_DIR = REPO_ROOT / "data" / "matches" / "raw"

OUT_DIR = REPO_ROOT / "data" / "lineups"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "assumed_lineup.csv"

ROSTER_TARGET = 15
CUM_MIN_FRACTION = 0.75
TEAM_TOTAL_MINUTES = 990
CAP_MINUTES = 90.0
EPS = 1e-9

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
    "FC Edmonton": "Edmonton"
}

def make_match_id(row: pd.Series) -> str:
    season = str(row.get("Season", "")).strip()
    date = str(row.get("Date", "")).strip()[:10]
    home = str(row.get("Hometeam", "")).strip().replace(" ", "_")
    away = str(row.get("Awayteam", "")).strip().replace(" ", "_")
    return f"{season}_{date}_{home}_vs_{away}"

def redistribute_with_cap(minutes: pd.Series, cap: float, total: float) -> pd.Series:
    m = minutes.astype(float).copy().clip(lower=0.0)
    if m.sum() < EPS:
        return m
    m *= (total / m.sum())
    for _ in range(50):
        over = (m - cap).clip(lower=0.0)
        overflow = over.sum()
        if overflow < 1e-6:
            break
        m = m.clip(upper=cap)
        under_mask = m < (cap - 1e-9)
        if not under_mask.any():
            break
        weights = m[under_mask].copy()
        if weights.sum() < EPS:
            weights[:] = 1.0
        m.loc[under_mask] += overflow * (weights / weights.sum())
        if abs(m.sum() - total) > 1e-3:
            m *= (total / m.sum())
    if m.sum() > EPS:
        m *= (total / m.sum())
    return m

def select_roster(team_players: pd.DataFrame) -> pd.DataFrame:
    tp = team_players.copy()
    tp["Minutes"] = pd.to_numeric(tp["Minutes"], errors="coerce").fillna(0.0)
    tp = tp.sort_values("Minutes", ascending=False)
    if len(tp) <= 11:
        return tp
    cum = tp["Minutes"].cumsum()
    cutoff = CUM_MIN_FRACTION * tp["Minutes"].sum()
    roster_mask = cum <= cutoff
    if roster_mask.sum() < 11:
        roster = tp.head(11).copy()
    else:
        roster = tp.loc[roster_mask].copy()
    return roster.head(ROSTER_TARGET)

def build_expected_minutes(roster: pd.DataFrame) -> pd.Series:
    mins = pd.to_numeric(roster["Minutes"], errors="coerce").fillna(500.0).astype(float)
    if mins.sum() < EPS:
        mins = pd.Series(1.0, index=roster.index)
    raw = TEAM_TOTAL_MINUTES * (mins / mins.sum())
    return redistribute_with_cap(raw, CAP_MINUTES, TEAM_TOTAL_MINUTES)

def read_all_matches(matches_dir: Path) -> pd.DataFrame:
    files = sorted(matches_dir.glob("matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No match files found in {matches_dir}")
    dfs = [pd.read_csv(f) for f in files]
    allm = pd.concat(dfs, ignore_index=True)
    allm.columns = [c.capitalize() for c in allm.columns]
    allm = allm.fillna("")
    if "Status" in allm.columns:
        allm = allm[allm["Status"].astype(str).str.upper() == "FINISHED"].copy()
    allm["match_id"] = allm.apply(make_match_id, axis=1)
    allm["Season"] = pd.to_numeric(allm["Season"], errors="coerce").fillna(0).astype(int)
    return allm

def run_validation(df: pd.DataFrame):
    valid_df = df[df["playerName"] != ""].copy()
    if valid_df.empty:
        print("Validation skipped: No valid player rows generated.")
        return
    stats = valid_df.groupby(["match_id", "team"]).agg(
        total_mins=("expected_minutes", "sum"),
        player_count=("playerName", "count")
    )
    mins_fail = stats[abs(stats["total_mins"] - 990) > 0.5]
    roster_fail = stats[stats["player_count"] < 11]
    print("\n" + "="*40)
    print("       AUTO-VALIDATION REPORT")
    print("="*40)
    if mins_fail.empty:
        print("MINUTE CHECK PASSED")
    else:
        print(f"MINUTE ERROR: {len(mins_fail)} teams failed.")
    if not roster_fail.empty:
        print(f"DATA GAP: {len(roster_fail)} teams have < 11 players.")
    else:
        print("ROSTER CHECK PASSED.")
    print(f"Avg Roster Size: {stats['player_count'].mean():.1f}")
    print("="*40 + "\n")

def main() -> None:
    if not PLAYER_BASE.exists():
        raise FileNotFoundError(f"Player base file not found: {PLAYER_BASE}")
    
    players = pd.read_csv(PLAYER_BASE)
    matches = read_all_matches(MATCHES_RAW_DIR)

    players["season"] = pd.to_numeric(players["season"], errors="coerce").fillna(0).astype(int)
    players["team"] = players["team"].astype(str).str.strip().replace(TEAM_NAME_MAP)
    players["playerName"] = players["playerName"].astype(str).str.strip()
    
    # Ensure playerId exists and is clean
    if "playerId" not in players.columns:
        # Fallback if your player base somehow lost its ID column
        players["playerId"] = players["playerName"].str.replace(" ", "_").str.lower()
    else:
        players["playerId"] = players["playerId"].astype(str).str.strip()

    matches["Season"] = pd.to_numeric(matches["Season"], errors="coerce").fillna(0).astype(int)
    for col in ["Hometeam", "Awayteam"]:
        matches[col] = matches[col].astype(str).str.strip().replace(TEAM_NAME_MAP)

    out_rows = []
    
    for _, m in matches.iterrows():
        s_val = int(m["Season"])
        d_val = str(m["Date"])
        mid_val = str(m["match_id"])
        
        for team_col in ["Hometeam", "Awayteam"]:
            t_raw = str(m[team_col]).strip()
            t_mapped = TEAM_NAME_MAP.get(t_raw, t_raw)
            
            tp = players[(players["season"] == s_val) & (players["team"] == t_mapped)].copy()
            
            if tp.empty:
                out_rows.append({
                    "match_id": mid_val, "season": s_val, "date": d_val, "team": t_raw,
                    "playerId": "MISSING", "playerName": "", "expected_minutes": 0.0, "source": "MISSING"
                })
                continue

            roster = select_roster(tp)
            expected = build_expected_minutes(roster)
            
            for idx, r in roster.iterrows():
                out_rows.append({
                    "match_id": mid_val, 
                    "season": s_val, 
                    "date": d_val, 
                    "team": t_raw,
                    "playerId": r["playerId"], # <--- NOW INCLUDED
                    "playerName": str(r["playerName"]),
                    "expected_minutes": round(float(expected.loc[idx]), 2),
                    "source": "calculated"
                })
                
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["season", "date", "match_id", "team", "expected_minutes"], ascending=[True, True, True, True, False])
    
    out.to_csv(OUT_FILE, index=False)
    print(f"Process Complete! Saved with playerId to: {OUT_FILE}")
    run_validation(out)

if __name__ == "__main__":
    main()