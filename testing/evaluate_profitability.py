import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

PREDS_FILE = REPO_ROOT / "data" / "matches" / "derived" / "james_ml_predictions.csv"
ELO_FILE   = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

# Betting Settings
STARTING_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MIN_EDGE = 0.02
BOOKIE_VIG = 0.05  

def simulate_bookie_odds(elo_diff: float):
    # This generates "Draw No Bet" style odds (2-way market)
    prob_home_fair = 1 / (1 + 10 ** ((-elo_diff) / 400))
    prob_away_fair = 1 - prob_home_fair

    # Apply overround
    prob_home_book = prob_home_fair * (1 + BOOKIE_VIG)
    prob_away_book = prob_away_fair * (1 + BOOKIE_VIG)
    
    # --- FIX: Prevent ZeroDivisionError ---
    # Ensure probability is never exactly 0.0 or 1.0 (epsilon cap)
    epsilon = 1e-9
    prob_home_book = max(epsilon, min(prob_home_book, 1 - epsilon))
    prob_away_book = max(epsilon, min(prob_away_book, 1 - epsilon))
    # --------------------------------------

    odds_home = 1 / prob_home_book
    odds_away = 1 / prob_away_book
    return odds_home, odds_away

def main():
    print("Profitability Evaluation (Draw No Bet)")

    if not PREDS_FILE.exists() or not ELO_FILE.exists():
        print("Missing input files. Run pipeline.")
        return

    preds = pd.read_csv(PREDS_FILE)
    elo   = pd.read_csv(ELO_FILE)

    # Grab ID and Probabilities from your ML Model
    preds_subset = preds[['match_id', 'prob_home', 'prob_away']]
    
    # Grab Date, Teams, and Results from the Master File
    elo_subset = elo[['match_id', 'date', 'home_team', 'away_team', 'diff_total', 'label']]
    
    # Merge
    df = pd.merge(preds_subset, elo_subset, on='match_id', how='inner')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    bankroll = STARTING_BANKROLL
    history = [bankroll]
    bets_won = 0
    bets_lost = 0
    bets_push = 0

    print(f"Starting bankroll: ${bankroll:.2f}")
    print(f"Simulating {len(df)} matches...")
    print("-" * 80)
    print(f"{'Date':<12} {'Match':<28} {'Bet':<6} {'Odds':<7} {'Edge':<7} {'Wager':<10} {'Result':<6}")
    print("-" * 80)

    for i, row in df.iterrows():
        # 1. Get 2-Way Odds
        odds_h, odds_a = simulate_bookie_odds(row['diff_total'])

        # 2. Renormalize XGBoost Probs to match 2-Way Market
        my_p_h_raw = float(row['prob_home'])
        my_p_a_raw = float(row['prob_away'])
        p_total = my_p_h_raw + my_p_a_raw
        
        if p_total == 0: 
            history.append(bankroll)
            continue
            
        my_p_h = my_p_h_raw / p_total
        my_p_a = my_p_a_raw / p_total

        # 3. Calculate Edge
        edge_h = (my_p_h * odds_h) - 1
        edge_a = (my_p_a * odds_a) - 1

        bet_side = None
        bet_odds = 0.0
        edge = 0.0
        my_prob = 0.0

        if edge_h > MIN_EDGE and edge_h > edge_a:
            bet_side = "HOME"
            bet_odds = odds_h
            edge = edge_h
            my_prob = my_p_h
        elif edge_a > MIN_EDGE and edge_a > edge_h:
            bet_side = "AWAY"
            bet_odds = odds_a
            edge = edge_a
            my_prob = my_p_a

        # 4. Calculate Kelly Wager
        wager = 0.0
        if bet_side:
            b = bet_odds - 1.0
            p = my_prob
            q = 1.0 - p
            # Avoid division by zero in Kelly too
            if b <= 0: 
                wager = 0
            else:
                full_kelly = ((b * p) - q) / b
                safe_kelly = full_kelly * KELLY_FRACTION
                wager = bankroll * safe_kelly

                # Cap max bet at 10%
                wager = min(wager, bankroll * 0.10)
                wager = max(0.0, wager)

        outcome = "SKIP"
        
        # 5. Resolve Bet
        if wager > 0:
            if row['label'] == 1:
                # DRAW = REFUND (Push)
                outcome = "PUSH"
                bets_push += 1
            elif (bet_side == "HOME" and row['label'] == 2) or (bet_side == "AWAY" and row['label'] == 0):
                # WIN
                bankroll += wager * (bet_odds - 1.0)
                bets_won += 1
                outcome = "WIN"
            else:
                # LOSS
                bankroll -= wager
                bets_lost += 1
                outcome = "LOSS"
                
            # Print first 5 and last 5 bets
            if i < 5 or i > len(df) - 5:
                 match_name = f"{row['home_team']} vs {row['away_team']}"[:24]
                 print(f"{row['date'].strftime('%Y-%m-%d'):<12} {match_name:<28} {bet_side:<6} {bet_odds:<7.2f} {edge:<7.1%} ${wager:<10.2f} {outcome}")

        history.append(bankroll)

    # Final Stats
    total_bets = bets_won + bets_lost + bets_push
    active_bets = bets_won + bets_lost 
    
    roi = ((bankroll - STARTING_BANKROLL) / STARTING_BANKROLL) * 100
    win_rate = (bets_won / active_bets) * 100 if active_bets > 0 else 0.0

    print("-" * 80)
    print("FINAL RESULTS")
    print(f"Bankroll:   ${bankroll:.2f}")
    print(f"Profit:     ${bankroll - STARTING_BANKROLL:.2f}")
    print(f"ROI:        {roi:.2f}%")
    print(f"Total Bets: {total_bets} (W:{bets_won} L:{bets_lost} P:{bets_push})")
    print(f"Win Rate:   {win_rate:.1f}% (Excluding Pushes)")

    out_img = REPO_ROOT / "data" / "matches" / "derived" / "profit_chart.png"
    plt.figure(figsize=(12, 6))
    plt.plot(history, label='Bankroll')
    plt.axhline(STARTING_BANKROLL, linestyle='--', label='Start')
    plt.title(f"Profitability vs Simulated Odds (ROI: {roi:.1f}%)")
    plt.xlabel('Matches')
    plt.ylabel('Bankroll $')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_img)
    print(f"Chart saved to: {out_img}")

if __name__ == "__main__":
    main()