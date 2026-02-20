CPL Predictor: Power Rankings and Historical Priors
This branch contains the core Predictive Engine for the 2026 CPL Season. The model utilizes a historical Z-Score normalization process to establish a baseline for every team before the season begins.

The Pipeline
The script power_rankings_pipeline.py automates the ingestion and processing of data from 2021–2025.

Fetch: Pulls raw match and team data.

Correlate: Identifies which stats (e.g., Goals from Open Play, Clean Sheets) most accurately predict Total Points.

Weight: Applies a decay rate to give more importance to recent seasons (2024/2025).

Rank: Generates a Strength_Score for every team.

Match Prediction Logic (The Magic 50)
Predictions are generated using a Poisson Distribution via predict_via_power_rankings.py.

Scaling and Calibration

Scaling Factor (S=50): We use a divisor of 50 to translate the statistical strength gap into an expected goal difference (lambda).

Example: A 25-point gap / 50 = 0.5 goal advantage.

Empirical Grounding: This factor was chosen based on 2025 Backtesting, where it achieved a 48.72% outcome accuracy and a Log Loss of 1.0210.

Neutral Bias: Currently, the model is Home Blind, meaning it calculates pure team strength without accounting for stadium advantage (HFA).

Next Steps: Ensemble Integration
The project is moving toward an Ensemble Model (80/20 Split):

80% Weight: Historical Power Rankings (this model).

20% Weight: Supplemental model (Teammate integration).
This approach balances long-term historical trends with short-term variables or specific team analysis.

Usage
To predict a 2026 matchup using team IDs:

Bash
python source/models/standard/power_rankings/predict_via_power_rankings.py [HomeID] [AwayID] 2026
