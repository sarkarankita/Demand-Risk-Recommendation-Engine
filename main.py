# main.py

from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.preprocessing import (
    load_raw_data,
    preprocess_datetime,
    build_hourly_demand
)

from src.demand_analysis import (
    log_transform,
    compute_baseline
)

from src.risk_engine import (
    compute_risk_scores,
    assign_risk_levels,
    aggregate_daily_risk
)

from src.recommendation_engine import (
    generate_recommendations
)

# ==================================================
# 🔧 CONFIG FLAGS (IMPORTANT)
# ==================================================
SIMULATE_DEMAND_SPIKE = True     # ← change to False for normal mode
SPIKE_MULTIPLIER = 2.0           # try 1.5 / 2.0 / 2.5

FORECAST_HOURS = 24              # 24 = next day, 168 = next 7 days
DATE_COLUMN = "Date"


def simulate_demand_spike(
    forecast: pd.Series,
    multiplier: float
) -> pd.Series:
    """
    Simulate demand surge (stress testing).
    """
    return forecast * multiplier


def main():
    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------
    DATA_PATH = (
        Path(__file__).parent
        / "data"
        / "raw"
        / "tokenized_access_logs.csv"
    )

    df = load_raw_data(DATA_PATH)
    df = preprocess_datetime(df, date_col=DATE_COLUMN)

    # --------------------------------------------------
    # 2. BUILD HOURLY DEMAND
    # --------------------------------------------------
    hourly_demand = build_hourly_demand(df, date_col=DATE_COLUMN)

    # --------------------------------------------------
    # 3. LOG TRANSFORM (MODEL ONLY)
    # --------------------------------------------------
    log_demand = log_transform(hourly_demand)

    # --------------------------------------------------
    # 4. BASELINE (BUSINESS SCALE ✅)
    # --------------------------------------------------
    baseline_mean, baseline_std = compute_baseline(hourly_demand)

    # --------------------------------------------------
    # 5. SARIMA FORECAST
    # --------------------------------------------------
    model = SARIMAX(
        log_demand,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    forecast_res = results.get_forecast(steps=FORECAST_HOURS)
    forecast_log = forecast_res.predicted_mean

    # Convert back to original scale
    forecast = np.expm1(forecast_log)

    # --------------------------------------------------
    # 🔥 6. SIMULATE DEMAND SURGE (OPTIONAL)
    # --------------------------------------------------
    if SIMULATE_DEMAND_SPIKE:
        forecast = simulate_demand_spike(
            forecast,
            multiplier=SPIKE_MULTIPLIER
        )

    # --------------------------------------------------
    # 7. RISK SCORING
    # --------------------------------------------------
    risk_df = compute_risk_scores(
        forecast=forecast,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std
    )

    risk_df = assign_risk_levels(risk_df)

    # --------------------------------------------------
    # 8. DAILY AGGREGATION
    # --------------------------------------------------
    daily_risk = aggregate_daily_risk(risk_df)

    # --------------------------------------------------
    # 9. BUSINESS RECOMMENDATIONS
    # --------------------------------------------------
    recommendations = generate_recommendations(daily_risk)

    print("\n=== FINAL RECOMMENDATIONS ===")
    print(recommendations)


if __name__ == "__main__":
    main()
