from __future__ import annotations
import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Example feature builder.
    Assumes df already contains date-sorted rows.
    """
    out = df.copy()
    # TODO: add rolling stats, rest days, elo diffs, etc.
    # Example: create a dummy constant feature to test the pipeline
    out["feat_bias"] = 1.0
    return out
