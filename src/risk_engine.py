import pandas as pd


def compute_risk_scores(forecast: pd.Series, baseline_mean: float, baseline_std: float) -> pd.DataFrame:
    """
    Convert forecast into z-score based risk levels.
    """
    if baseline_std == 0:
        baseline_std = 1e-6

    df = forecast.to_frame(name="forecast_demand")
    df["z_score"] = (df["forecast_demand"] - baseline_mean) / baseline_std
    return df


def classify_risk(z_score: float) -> str:
    if z_score >= 2:
        return "HIGH"
    elif z_score >= 1:
        return "MEDIUM"
    else:
        return "LOW"


def assign_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level"] = df["z_score"].apply(classify_risk)
    return df


def aggregate_daily_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df.index).date

    daily = (
        df.groupby("date")
        .agg(
            avg_demand=("forecast_demand", "mean"),
            max_demand=("forecast_demand", "max"),
            high_risk_hours=("risk_level", lambda x: (x == "HIGH").sum()),
            medium_risk_hours=("risk_level", lambda x: (x == "MEDIUM").sum()),
        )
    )
    return daily
