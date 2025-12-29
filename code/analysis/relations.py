import pandas as pd

# ----------------------------
# Paths (relative to project root)
# ----------------------------
input_csv = 'data/teams/combined/teams_combined.csv'
output_csv = 'data/teams/combined/stat_correlations.csv'

# ----------------------------
# Load the combined teams data
# ----------------------------
df_teams_all = pd.read_csv(input_csv)

# ----------------------------
# Analyze team stats
# ----------------------------
correlations = df_teams_all.corr(numeric_only=True)['Total points'].sort_values(ascending=False)

print("\n--- TOP 10 STATS THAT LEAD TO WINS ---")
print(correlations.head(100))

# ----------------------------
# Save full correlation list
# ----------------------------
correlations.to_csv(output_csv, sep='\t')
print(f"\nFull correlation list saved to: {output_csv}")
