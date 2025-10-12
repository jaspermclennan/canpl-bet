from __future__ import annotations
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

def evaluate_probs(y_true, p_true) -> dict:
    return {
        "log_loss": float(log_loss(y_true, p_true)),
        "brier": float(brier_score_loss(y_true, p_true)),
    }
