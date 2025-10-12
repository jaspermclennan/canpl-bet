from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def fit_calibrated_logit(X: pd.DataFrame, y) -> CalibratedClassifierCV:
    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X, y)
    return model
