from __future__ import annotations
import pandas as pd

def chronological_split(df: pd.DataFrame, frac: float = 0.8):
    df = df.sort_values("date")
    n = int(len(df) * frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()
