import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

DATA_PATH = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"
PLOT_DIR = REPO_ROOT / "data" / "matches" / "derived"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "predicted_vs_actual_tables.png"

def calculate_points(predictions, actuals, teams_home, teams_away):
    """
    aggregates points for a list of matches.
    predictions/actuals: 0 (Away), 1 (Draw), 2 (Home)
    """
    table = {}
    
    # Initialize
    all_teams = set(teams_home) | set(teams_away)
    for t in all_teams:
        table[t] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'Pts': 0}
        
    for i in range(len(predictions)):
        h = teams_home[i]
        a = teams_away[i]
        pred = predictions[i]
        
        # 0=Away Win, 1=Draw, 2=Home Win
        if pred == 2: # Home Win
            table[h]['W'] += 1; table[h]['Pts'] += 3
            table[a]['L'] += 1
        elif pred == 1: # Draw
            table[h]['D'] += 1; table[h]['Pts'] += 1
            table[a]['D'] += 1; table[a]['Pts'] += 1
        elif pred == 0: # Away Win
            table[h]['L'] += 1
            table[a]['W'] += 1; table[a]['Pts'] += 3
            
        table[h]['P'] += 1
        table[a]['P'] += 1
        
    return pd.DataFrame.from_dict(table, orient='index').sort_values('Pts', ascending=False)

def main():
    if not DATA_PATH.exists():
        print(f"❌ Missing {DATA_PATH}. Run build_rolling_features.py first.")
        return

    print("--- 🏆 GENERATING PREDICTED LEAGUE TABLES (2022-2025) ---")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Storage for analysis
    full_history = []
    
    # Burn-in
    START_INDEX = 50
    
    # 1. GENERATE ALL PREDICTIONS (WALK-FORWARD)
    print(f"   Simulating {len(df) - START_INDEX} matches...")
    
    for i in range(START_INDEX, len(df)):
        train_df = df.iloc[:i]
        current_match = df.iloc[[i]]
        
        # Features
        features = ['diff_total', 'diff_form_pts', 'diff_form_gd']
        X_train = train_df[features].values
        y_train = train_df['label'].values
        X_test = current_match[features].values
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model
        model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        pred_label = model.predict(X_test_scaled)[0]
        
        full_history.append({
            'season': current_match['season'].values[0],
            'home': current_match['home_team'].values[0],
            'away': current_match['away_team'].values[0],
            'actual_label': current_match['label'].values[0],
            'pred_label': pred_label
        })
        
    history_df = pd.DataFrame(full_history)
    
    # 2. BUILD TABLES PER SEASON
    seasons = sorted(history_df['season'].unique())
    
    # Prepare Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, season in enumerate(seasons):
        if idx >= 4: break # Only plot first 4 if more exist
        
        s_data = history_df[history_df['season'] == season]
        
        # Calculate Real Table
        real_table = calculate_points(
            s_data['actual_label'].values, 
            s_data['actual_label'].values, # Pass actual as both to calc real pts
            s_data['home'].values, 
            s_data['away'].values
        )
        
        # Calculate Predicted Table
        pred_table = calculate_points(
            s_data['pred_label'].values, 
            s_data['actual_label'].values,
            s_data['home'].values, 
            s_data['away'].values
        )
        
        # Merge for Comparison
        merged = real_table[['Pts']].rename(columns={'Pts': 'Actual_Pts'}).join(
            pred_table[['Pts']].rename(columns={'Pts': 'Pred_Pts'})
        )
        merged['Error'] = merged['Pred_Pts'] - merged['Actual_Pts']
        merged = merged.sort_values('Actual_Pts', ascending=False)
        
        # Print Table
        print(f"\nExample Season: {season}")
        print(f"{'Team':<20} | {'Act Pts':<8} | {'Pred Pts':<8} | {'Diff':<5}")
        print("-" * 50)
        for team, row in merged.iterrows():
            print(f"{team:<20} | {row['Actual_Pts']:<8} | {row['Pred_Pts']:<8} | {row['Error']:<5}")

        # Plot Scatter
        ax = axes[idx]
        ax.scatter(merged['Actual_Pts'], merged['Pred_Pts'], color='blue', s=100)
        
        # Add team labels
        for team, row in merged.iterrows():
            ax.annotate(team, (row['Actual_Pts'], row['Pred_Pts']), xytext=(5, 5), textcoords='offset points')
            
        # Perfect prediction line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_title(f"{season} Season")
        ax.set_xlabel("Actual Points")
        ax.set_ylabel("Predicted Points")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"\n📊 Comparison Plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()