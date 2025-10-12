import argparse
import pandas as pd

def decimal_odds_from_american(odds):
    if odds >= 100:
        return 1 + odds/100.0
    else:
        return 1 + 100.0/abs(odds)

def kelly_fraction(p, O, kelly_frac=1.0):
    # Full Kelly default; multiply by 0.25 for quarter-Kelly at call site
    edge = p*(O-1) - (1-p)
    if O <= 1 or edge <= 0:
        return 0.0
    return kelly_frac * edge/(O-1)

def main(args):
    # expects CSV with columns: pred (prob of event), odds (american), result (0/1)
    df = pd.read_csv(args.csv)
    O = df["odds"].apply(decimal_odds_from_american)
    f = [kelly_fraction(p, o, kelly_frac=args.kelly) for p, o in zip(df["pred"], O)]
    bankroll = args.bankroll
    curve = [bankroll]
    for frac, p, o, r in zip(f, df["pred"], O, df["result"]):
        stake = bankroll * frac
        pnl = stake*(o-1) if r == 1 else -stake
        bankroll += pnl
        curve.append(bankroll)
    print(f"Final bankroll: {bankroll:.2f}")
    pd.DataFrame({"bankroll": curve}).to_csv("artifacts/bankroll_curve.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns pred,odds,result")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier, e.g., 0.25 for quarter-Kelly")
    main(ap.parse_args())
