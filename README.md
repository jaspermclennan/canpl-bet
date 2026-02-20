# Canadian Premier League Predictor: Power Rankings and Match Prediction with Historical Priors

This branch contains the core Predictive Engine for the 2026 CPL Season. The model utilizes a historical z-score normalization process to establish a baseline power ranking for every team before the season begins. This model is tuned to predict the first round of matches in the CPL's 2026 season, taking into account statistics from seasons 2021-2025 at a decaying rate. Due to changes in league data collection, as well as high squad turnover, we do not account for or utilize the 2019 or 2020 seasons in our model.

## The Pipeline
The script `power_rankings_pipeline.py` automates the intake and processing of data from 2021-2025. This script combines the below steps into one. If desired, a user can run each of these steps individually from the corresponding files.
1. **Fetch (source/fetch_data):** Pulls raw match and team data from Canadian Premier League API into csv files by year.
2. **Aggregate (source/process_data/aggregation):** Combines csv files of yearly team and match data into cleaned csv files.
3. **Correlate (source/process_data/pre_model):** Identifies which stats (e.g., Goals from Open Play, Clean Sheets) most accurately predict Total Points using .corr() method in Pandas.
4. **Standardize (source/process_data/pre_model):** Calculates standardized strength scores for every team in every season by applying correlation-weighted z-scores to significant performance metrics (from correlations above), creating a historical baseline for match prediction by ranking each team across all seasons.
5. **Rank (source/process_data/pre_model):** Generates and saves to csv a final strength score for every team to serve as the "Historical Prior" for the upcoming season we wish to predict.

## Match Prediction Logic
Once the above pipeline has been completed, predictions can be generated using a **Poisson Distribution** via `predict_via_power_rankings.py`. Specifications on how to use the prediction script can be seen in the "Usage" section below.

### Scaling and Calibration
* **Scaling Factor (S=50):** We use a divisor of 50 to translate the statistical strength gap into an expected goal difference (lambda).
  * *Example:* A 25-point gap / 50 = 0.5 goal advantage.
* **Neutral Bias:** Currently, the model is **Home Blind**, meaning it calculates pure team strength without accounting for stadium advantage (HFA).

### Testing and Validation
* **Empirical Grounding:** This factor was chosen based on **2025 Backtesting**, where it achieved a **48.72% outcome accuracy** and a Log Loss of **1.0210**.

## Next Steps: Ensemble Integration and Machine Learning
The first segment of this project is moving toward an **Ensemble Model (X/Y/Z Split)**:
* **X% Weight:** Historical Power Rankings (this model).
* **Y% Weight:** Supplemental model based on players involved in a given match (Teammate integration).
* **Z% Weight:** External factors such as team fatigue, weather, home advantage, injuries, importance of game.

## Usage
To predict a 2026 opening round matchup using team IDs:

1) Run power_rankings_pipeline.py with a specified target year to fetch and manipulate all relevant data. 

2) Run`python source/models/standard/power_rankings/predict_via_power_rankings.py [HomeID] [AwayID] 2026`

       (IDs: 1: Ottawa, 2: Forge, 3: Cavalry, 4: HFX, 5: York/Toronto, 6: Pacific, 7: Vancouver, 8: Quebec)
