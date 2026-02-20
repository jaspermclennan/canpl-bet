# @AUTHOUR: Jasper McLennan
import os

def main():
    # Central control for the entire project
    target_year = "2025"
    
    print(f"--- CPL PIPELINE: {target_year} PRODUCTION RUN ---")

    scripts = [
        "source/fetch_data/match_data.py",
        "source/fetch_data/team_data.py",
        "source/process_data/aggregation/combine_match_data.py",
        "source/process_data/aggregation/combine_team_data.py",
        "source/process_data/pre_model/correlations.py",
        "source/process_data/pre_model/historical_team_power_rankings.py",
        "source/process_data/pre_model/predicted_strengths.py"
    ]

    for script in scripts:
        print(f"\n[STEP]: {script}")
        # Passing the year here ensures NO accidents
        os.system(f"python {script} {target_year}")

    print("\n" + "="*40)
    print(f" DONE: {target_year} BASELINE IS BUILT")
    print("="*40)

if __name__ == "__main__":
    main()