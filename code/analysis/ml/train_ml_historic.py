import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load the training set you just created
df = pd.read_csv('data/analysis/ml_training_priors.csv')

# 2. Prepare the Features
# The model learns from the 'Gap' between the two teams
df['Prior_Gap'] = df['Home_Prior'] - df['Away_Prior']

# Drop rows where Priors are missing (e.g., if a team name didn't match perfectly)
df = df.dropna(subset=['Prior_Gap', 'Result'])

# X is our input (the gap), y is our target (the result)
X = df[['Prior_Gap']]
y = df['Result'] # 0: Home Win, 1: Draw, 2: Away Win

# 3. Train the Model
# 'multinomial' handles 3+ outcomes, 'lbfgs' is a standard fast solver
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)

# 4. Save the "Brain"
# This saves the trained mathematical weights so you don't have to retrain every time
joblib.dump(model, 'data/analysis/cpl_ml_model.pkl')

print("--- ML MODEL TRAINED ---")
print(f"Studied {len(df)} historical matches.")
print("Model saved to: data/analysis/cpl_ml_model.pkl")