CPL Predictor ⚽️📈
A data-driven engine for the Canadian Premier League.

This is a personal project we built to see if we could use math and machine learning to beat sports markets at predicting CPL matches. It doesn't just look at who won the last game; it looks at years of team history and goal-scoring patterns to find "Value Bets."

🕹️ How it Works
We use two different "brains" to make a prediction:

The Math Brain (Poisson): It looks at how many goals teams usually score and calculate the odds of every possible scoreline (1-0, 2-2, etc.).

The Machine Brain (Logistic Regression): It has studied over 500 historical CPL matches to learn how a "Strength Gap" between two teams actually translates into a Win, Loss, or Draw.

🛠️ Key Features
Backtesting: I built a "time machine" script that tests my models against the entire 2025 season to see which one was actually more accurate.

Neutral Odds: The models are trained to be "neutral," so I can manually add a Home Field Advantage (HFA) bonus depending on where the game is being played.

Automated Pipeline: The code automatically cleans up messy CSV data and standardizes team names and years.

🚀 Quick Start
To get the odds for a 2026 matchup (e.g., Cavalry vs. Forge):



python3 code/analysis/master_odds.py 2026 1 2



🚧 What’s Next (In Progress)
The engine is currently getting a few upgrades to move from "Team-level" to "Game-level" accuracy:

Player Values: Integrating individual player stats and market values to account for roster changes and key injuries.

Travel & Fatigue: Tracking travel distance (CPL is a huge country!) and "days since last match" to see how they impact performance.

External Factors: Pulling in weather data to adjust expected goal counts for rainy or high-wind matchdays.
