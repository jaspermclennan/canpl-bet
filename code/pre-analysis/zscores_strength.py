import pandas as pd

# 1. Load Data
df_teams = pd.read_csv('data/teams/combined/teams_combined.csv')
df_corr = pd.read_csv('data/analysis/correlations.csv', sep='\t', index_col=0)

# 2. Automated Selection: Find stats with absolute correlation > 0.3
threshold = 0.25
significant_stats = df_corr[df_corr['Total points'].abs() > threshold].index.tolist()

# Clean list of stats to process
exclude_cols = ['Total points', 'Games played', 'Year', 'Team']
significant_stats = [s for s in significant_stats if s not in exclude_cols]

# 3. Calculate Z-Scores & Weighted Strength
zscore_frames = []

for year, season_df in df_teams.groupby("Year"):
    # Start with basic info
    season_z = season_df[['Team', 'Year', 'Total points']].copy()
    weighted_columns = []
    
    for stat in significant_stats:
        if stat in season_df.columns:
            mean = season_df[stat].mean()
            std = season_df[stat].std()
            
            # Standardize (Z-Score)
            z_col = (season_df[stat] - mean) / std if std != 0 else 0
            
            # Apply Weight (Correlation)
            weight = df_corr.loc[stat, 'Total points']
            col_name = f"{stat}_w"
            season_z[col_name] = z_col * weight
            weighted_columns.append(col_name)

    # Sum all weighted stats into one Score
    season_z['Strength_Score'] = season_z[weighted_columns].sum(axis=1)
    
    # ---------------------------------------------------------
    # THE CLEANUP: Drop the individual weighted stat columns
    # ---------------------------------------------------------
    season_z = season_z[['Team', 'Year', 'Total points', 'Strength_Score']]
    
    zscore_frames.append(season_z)

# 4. Save Final Team Profiles
df_strength = pd.concat(zscore_frames, ignore_index=True)

# Sort by Strength so you can see the power ranking immediately
df_strength = df_strength.sort_values(by=['Year', 'Strength_Score'], ascending=[False, False])

df_strength.to_csv('data/analysis/team_strengths.csv', index=False)

print("\n--- FINAL TEAM STRENGTH RANKINGS ---")
print(df_strength.head(80))
print(f"\nSaved clean table to: data/analysis/team_strengths.csv")