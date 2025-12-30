import pandas as pd

# 1. CONFIG
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

# --- CLEANING AMENDMENT ---
# Remove leading/trailing whitespace from column names and team names
df.columns = df.columns.str.strip()
if 'Team' in df.columns:
    df['Team'] = df['Team'].str.strip()

# CALCULATE WEIGHTS
prev_years = sorted([y for y in df['Year'].unique() if y < TARGET_YEAR], reverse=True)

raw_w = [DECAY_RATE**i for i in range(len(prev_years))]
norm_w = [w / sum(raw_w) for w in raw_w]
weight_map = dict(zip(prev_years, norm_w))

# 4. APPLY WEIGHTS & COLLAPSE
hist_df = df[df['Year'] < TARGET_YEAR].copy()
hist_df['weighted_val'] = hist_df['Year'].map(weight_map) * hist_df['Strength_Score']
priors = hist_df.groupby('Team')['weighted_val'].sum().reset_index()

# --- NAME PRINTING AMENDMENT ---
# Dynamically name the column based on the TARGET_YEAR
prior_col_name = f'Historical Prior for {TARGET_YEAR} Season'
priors.columns = ['Team', prior_col_name]

# 5. FILTER FOR ACTIVE ROSTER
active_teams = LEAGUE_HISTORY.get(TARGET_YEAR, [])
priors = priors[priors['Team'].isin(active_teams)]

# 6. DATA-DRIVEN EXPANSION HANDLER
new_teams = [t for t in active_teams if t not in priors['Team'].values]

if new_teams:
    # Identify debut strength of all teams that joined AFTER the first season
    debut_data = df.sort_values('Year').groupby('Team').first().reset_index()
    expansion_debuts = debut_data[debut_data['Year'] > 2019]['Strength_Score']
    
    # Average debut strength (fallback to -15.0 if no history exists)
    expansion_prior = expansion_debuts.mean() if not expansion_debuts.empty else -15.0

    for team in new_teams:
        new_row = pd.DataFrame({'Team': [team], prior_col_name: [expansion_prior]})
        priors = pd.concat([priors, new_row], ignore_index=True)

# 7. OUTPUT
priors = priors.sort_values(by=prior_col_name, ascending=False)

# --- CSV CLEANING/NAME AMENDMENT ---
# Use dynamic filename for saving
output_path = f'data/analysis/predict_{TARGET_YEAR}_from_historic.csv'
priors.to_csv(output_path, index=False)

print(f"\n--- GENERATED {TARGET_YEAR} PRE-SEASON PRIORS ---")
print(priors.to_string(index=False))
print(f"\nResults saved to: {output_path}")