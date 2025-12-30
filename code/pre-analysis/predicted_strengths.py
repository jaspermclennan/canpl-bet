import pandas as pd
import sys

# ----------------------------
# 1. DYNAMIC CONFIG
# ----------------------------
if len(sys.argv) > 1:
    TARGET_YEAR = int(sys.argv[1])
else:
    TARGET_YEAR = 2026

DECAY_RATE = 0.3

# 2. ROSTERS (Participating teams per year)
LEAGUE_HISTORY = {
    2022: ['Forge', 'Cavalry', 'Atlético Ottawa', 'HFX Wanderers', 'York United', 'Pacific', 'Valour', 'Edmonton'],
    2023: ['Forge', 'Cavalry', 'Atlético Ottawa', 'HFX Wanderers', 'York United', 'Pacific', 'Valour', 'Vancouver FC'],
    2024: ['Forge', 'Cavalry', 'Atlético Ottawa', 'HFX Wanderers', 'York United', 'Pacific', 'Valour', 'Vancouver FC'],
    2025: ['Forge', 'Cavalry', 'Atlético Ottawa', 'HFX Wanderers', 'York United', 'Pacific', 'Valour', 'Vancouver FC'],
    2026: ['Forge', 'Cavalry', 'Atlético Ottawa', 'HFX Wanderers', 'York United', 'Pacific', 'Vancouver FC', 'FC Supra du Québec']
}

# 3. LOAD DATA & CLEANING
df = pd.read_csv('data/analysis/team_strengths.csv')

df.columns = df.columns.str.strip()
if 'Team' in df.columns:
    df['Team'] = df['Team'].str.strip()

# CALCULATE WEIGHTS
prev_years = sorted([y for y in df['Year'].unique() if y < TARGET_YEAR], reverse=True)

if not prev_years:
    print(f"Error: No historical data found prior to {TARGET_YEAR}.")
    sys.exit(1)

raw_w = [DECAY_RATE**i for i in range(len(prev_years))]
norm_w = [w / sum(raw_w) for w in raw_w]
weight_map = dict(zip(prev_years, norm_w))

# 4. APPLY WEIGHTS & COLLAPSE
hist_df = df[df['Year'] < TARGET_YEAR].copy()
hist_df['weighted_val'] = hist_df['Year'].map(weight_map) * hist_df['Strength_Score']
priors = hist_df.groupby('Team')['weighted_val'].sum().reset_index()

prior_col_name = f'Historical Prior for {TARGET_YEAR} Season'
priors.columns = ['Team', prior_col_name]

# 5. FILTER FOR ACTIVE ROSTER
active_teams = LEAGUE_HISTORY.get(TARGET_YEAR, [])
priors = priors[priors['Team'].isin(active_teams)]

# ----------------------------
# 6. ENHANCED EXPANSION HANDLER (Blended Logic)
# ----------------------------
new_teams = [t for t in active_teams if t not in priors['Team'].values]

if new_teams:
    # A. Calculate the League-Wide Average across all historical data
    # This represents the "True Neutral" of the CPL
    league_average_all_time = df['Strength_Score'].mean()

    # B. Identify debut strength of previous expansion teams
    debut_data = df.sort_values('Year').groupby('Team').first().reset_index()
    # Expansion = first appearance was after the league's 'original' data year (2022)
    # AND happened before our target year (to avoid leakage)
    expansion_performances = debut_data[
        (debut_data['Year'] > 2022) & 
        (debut_data['Year'] < TARGET_YEAR)
    ]['Strength_Score']
    
    if not expansion_performances.empty:
        expansion_mean = expansion_performances.mean()
        # C. The "Blended Prior": 70% Expansion History + 30% League Average
        # This pulls the score from -54 up toward 0, landing it in a more realistic "struggling" zone.
        expansion_prior = (expansion_mean * 0.7) + (league_average_all_time * 0.3)
        

    else:
        # Fallback if no expansion data exists yet (e.g., backtesting 2023)
        expansion_prior = -15.0
        print(f"No previous expansion data. Using baseline: {expansion_prior}")

    for team in new_teams:
        new_row = pd.DataFrame({'Team': [team], prior_col_name: [expansion_prior]})
        priors = pd.concat([priors, new_row], ignore_index=True)
        
# 7. OUTPUT
priors = priors.sort_values(by=prior_col_name, ascending=False)
output_path = f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv'
priors.to_csv(output_path, index=False)

print(f"\n--- EXPECTED {TARGET_YEAR} TEAM STRENGTH ---")
print(priors.to_string(index=False))
print(f"\nResults saved to: {output_path}")