import pandas as pd

# ----------------------------
# Paths
# ----------------------------
input_csv = 'data/teams/combined/teams_combined.csv'
output_csv = 'data/analysis/correlations.csv'

# ----------------------------
# Load Data
# ----------------------------
df_teams_all = pd.read_csv(input_csv)

# ----------------------------
# Define "Leakage" stats to exclude
# ----------------------------
exclude_stats = [
    'Total points', 'Total wins', 'Total losses', 'Total draws', 
    'Points', 'Wins', 'Losses', 'Draws', 
    'Games played', 'Games Played', 'Year', "Goals against"
]

# ----------------------------
# Analyze Predictive Stats
# ----------------------------
# 1. Calculate all correlations first
all_corrs = df_teams_all.corr(numeric_only=True)['Total points']

# 2. THE FIX: Drop the NaNs! 
# This removes stats where the league has missing data for certain years.
predictive_corrs = all_corrs.dropna()

# 3. Drop the "leakage" ones from the index
predictive_corrs = predictive_corrs.drop(labels=exclude_stats, errors='ignore').sort_values(ascending=False)

print("\n--- TOP 20 PREDICTIVE PERFORMANCE STATS ---")
print(predictive_corrs.head(20).to_frame())

print("\n--- TOP 20 PREDICTIVE NEGATIVE PERFORMANCE STATS ---")
# Sorting ascending=True shows the most damaging (negative) stats first
print(predictive_corrs.sort_values(ascending=True).head(20).to_frame())

# ----------------------------
# Save the Clean List
# ----------------------------
predictive_corrs.to_csv(output_csv, sep='\t')
print(f"\nCleaned correlation list (no NaNs) saved to: {output_csv}")