import numpy as np
import pandas as pd


def log_transform(series: pd.Series) -> pd.Series:
    """
    Apply log(1 + x) transform to stabilize variance.
    """
    return np.log1p(series)


def compute_baseline(series: pd.Series, window_hours: int = 24 * 7) -> tuple[float, float]:
    """
    Compute baseline mean and std from recent history.
    """
    recent = series[-window_hours:]
    mean = recent.mean()
    std = recent.std()

    if std == 0 or pd.isna(std):
        std = 1e-6

    return mean, std
