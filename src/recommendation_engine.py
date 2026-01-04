import pandas as pd


def classify_day_risk(row: pd.Series) -> str:
    if row["high_risk_hours"] >= 4:
        return "CRITICAL"
    elif row["medium_risk_hours"] >= 6:
        return "ELEVATED"
    else:
        return "NORMAL"


def recommend_action(day_risk: str) -> str:
    if day_risk == "CRITICAL":
        return "Increase capacity and prioritize deliveries"
    elif day_risk == "ELEVATED":
        return "Prepare buffer and monitor closely"
    else:
        return "No action required"


def generate_recommendations(daily_risk_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_risk_df.copy()
    df["day_risk"] = df.apply(classify_day_risk, axis=1)
    df["recommendation"] = df["day_risk"].apply(recommend_action)
    return df
