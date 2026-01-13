import pandas as pd
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet" else Path(__file__).resolve().parent.parent.parent.parent

DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"

def main():
    print("Tuning XGBoost Factors")
    
    if not DATA_FILE.exists():
        print(f"File not Found: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Feature selection (Add more???)
    features = [
        'diff_total', # ELO gap
        'diff_form_pts', 
        'diff_form_gd'
    ]
    target = 'label'
    
    # 2. Split data (should make moving date current)
    split_date = '2025-01-01'
    train_df = df[df['date'] < split_date].copy().dropna(subset=[target])
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    print(f"Tuning on {len(X_train)} matches before 2025")
    
    # 3. settings we want to test
    param_grid = {
        'max_depth': [2, 3, 4, 5], # complex tree
        'learning_rate': [0.01, 0.05, 0.1], # fast model learns quicker likely better
        'n_estimators': [50, 100, 200], #trees to build number of
        'subsample': [0.8, 1.0], # fraction of matches to use per tree
        'colsample_bytree': [0.8, 1.0], # fraction of features to use per tree
        'gamma': [0, 1, 5] #minimum loss reduction required to make a split
    }
    
    # 4. model setup
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )
    
    # Cross validation setup so we dont use the future to predict the past
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_log_loss', #closer to 0 the better
        cv=tscv,
        verbose=1,
        n_jobs=-1 #all CPU scores to be used
    )
    
    # 6. Run the search
    print("Running Grid Search")
    grid_search.fit(X_train, y_train)
    
    # 7. output results
    print("\n" + '=' * 40)
    print("Best Paramteres Found")
    best_params = grid_search.best_params_
    for param, value in best_params.items():
        print(f"   {param:<20}: {value}")
        
    print("-" * 40)
    print(f"   Best Log Loss Score: {-grid_search.best_score_:.4f}")
    
    # 8. Verification (Test on 2025 Data)
    print("\n VERIFICATION ON 2025 DATA")
    test_df = df[df['date'] >= split_date].copy().dropna(subset=[target])
    X_test = test_df[features]
    y_test = test_df[target]
    
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs)
    
    print(f"   Accuracy: {acc:.1%}")
    print(f"   Log Loss: {loss:.4f}")

if __name__ == "__main__":
    main()