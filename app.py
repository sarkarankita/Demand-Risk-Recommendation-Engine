
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

#IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from prophet import Prophet
from prophet.make_holidays import make_holidays_df

from src.preprocessing import load_raw_data, preprocess_datetime, build_hourly_demand
from src.demand_analysis import compute_baseline
from src.risk_engine import compute_risk_scores, assign_risk_levels, aggregate_daily_risk
from src.recommendation_engine import generate_recommendations

# PAGE CONFIG
st.set_page_config(page_title="Supply Chain Risk Intelligence", layout="wide")

# THEME
st.markdown("""
<style>
.stApp { background:#0b1220; color:#e5e7eb; }
.kpi-title { color:#38bdf8; font-weight:600; font-size:16px; }
.section { border-left:4px solid #38bdf8; padding-left:12px; }
</style>
""", unsafe_allow_html=True)

# TITLE
st.title(" Supply Chain Demand & Risk Intelligence Dashboard")
st.caption("Decision-focused forecasting, risk assessment & operational recommendations")

# LOAD DATA 
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tokenized_access_logs.csv"
raw_df = preprocess_datetime(load_raw_data(DATA_PATH), "Date")

min_date = raw_df["Date"].min().date()
max_date = raw_df["Date"].max().date()

# SIDEBAR 
st.sidebar.header(" Controls")

search = st.sidebar.text_input("Search Product")
products = sorted(raw_df["Product"].dropna().unique())

if search:
    products = [p for p in products if search.lower() in p.lower()]

selected_product = st.sidebar.selectbox(
    "Select Product",
    ["All Products"] + products
)

date_range = st.sidebar.date_input(
    "Historical Window",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if not isinstance(date_range, tuple) or date_range[0] >= date_range[1]:
    st.error(" Invalid date range")
    st.stop()

start_date, end_date = date_range
future_end = st.sidebar.date_input("Forecast Until", value=date(2026, 12, 31))

run = st.sidebar.button(" Run Analysis")
if not run:
    st.info("Configure inputs and click **Run Analysis**")
    st.stop()

# FILTER DATA 
df = raw_df[
    (raw_df["Date"].dt.date >= start_date) &
    (raw_df["Date"].dt.date <= end_date)
]

if selected_product != "All Products":
    df = df[df["Product"] == selected_product]

# PROPHET
def run_prophet(hourly, horizon):
    prophet_df = hourly.reset_index().rename(
        columns={"Date": "ds", hourly.name: "y"}
    )

    holidays = make_holidays_df(
        year_list=list(range(start_date.year, future_end.year + 1)),
        country="IN"
    )

    model = Prophet(
        holidays=holidays,
        daily_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="H")
    forecast = model.predict(future)

    return forecast.set_index("ds")["yhat"]

#  MAIN PIPELINE 
with st.spinner("Forecasting demand & computing risks..."):

    hourly_demand = build_hourly_demand(df, "Date")
    baseline_mean, baseline_std = compute_baseline(hourly_demand)

    horizon = max(
        24,
        (pd.to_datetime(future_end) - pd.to_datetime(end_date)).days * 24
    )

    forecast = run_prophet(hourly_demand, horizon)

    risk_df = assign_risk_levels(
        compute_risk_scores(forecast, baseline_mean, baseline_std)
    )

    daily_risk = aggregate_daily_risk(risk_df)
    decision = generate_recommendations(daily_risk).iloc[0]

# EXECUTIVE KPIs
st.markdown("## Executive Decision Summary", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='kpi-title'>Average Demand</div>", unsafe_allow_html=True)
    st.metric("", int(decision["avg_demand"]))

with c2:
    st.markdown("<div class='kpi-title'>Peak Demand</div>", unsafe_allow_html=True)
    st.metric("", int(decision["max_demand"]))

with c3:
    st.markdown("<div class='kpi-title'>Risk Level</div>", unsafe_allow_html=True)
    st.metric("", decision["day_risk"])

with c4:
    st.markdown("<div class='kpi-title'>Worst-Case Demand</div>", unsafe_allow_html=True)
    st.metric("", int(decision["max_demand"] * 1.15))

# FORECAST PLOT 
st.markdown("## Demand Forecast vs Historical", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hourly_demand.tail(72).index,
    y=hourly_demand.tail(72),
    name="Historical"
))
fig.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast,
    name="Forecast"
))

fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

# DECISION TABLE 
st.markdown("## Decision Table", unsafe_allow_html=True)

decision_table = pd.DataFrame([
    ["Scope", selected_product],
    ["Avg Demand", round(decision["avg_demand"], 0)],
    ["Peak Demand", round(decision["max_demand"], 0)],
    ["High Risk Hours", decision["high_risk_hours"]],
    ["Risk Level", decision["day_risk"]],
    ["Recommended Action", decision["recommendation"]],
], columns=["Metric", "Value"])

st.dataframe(decision_table, use_container_width=True)

# WORST CASE 
st.markdown("## Worst-Case Scenario", unsafe_allow_html=True)

st.warning(
    f"If no action is taken, demand may exceed baseline by "
    f"**{decision['high_risk_hours']} hours**, potentially causing "
    f"**inventory shortages, delayed shipments, and SLA breaches**."
)

# TOP RISKY PRODUCTS 
st.markdown("## Top Risky Products", unsafe_allow_html=True)

rows = []
for p in raw_df["Product"].dropna().unique():

    temp = raw_df[
        (raw_df["Product"] == p) &
        (raw_df["Date"].dt.date >= start_date) &
        (raw_df["Date"].dt.date <= end_date)
    ]

    if len(temp) < 48:
        continue

    hourly = build_hourly_demand(temp, "Date")
    mean, std = compute_baseline(hourly)
    fc = run_prophet(hourly, 24)

    rdf = assign_risk_levels(compute_risk_scores(fc, mean, std))
    d = generate_recommendations(aggregate_daily_risk(rdf)).iloc[0]

    rows.append({
        "Product": p,
        "Avg Demand": round(d["avg_demand"], 0),
        "Peak Demand": round(d["max_demand"], 0),
        "Risk Hours": d["high_risk_hours"],
        "Risk Level": d["day_risk"]
    })

top_products = (
    pd.DataFrame(rows)
    .sort_values(["Risk Level", "Risk Hours"], ascending=[False, False])
    .head(5)
)

st.dataframe(top_products, use_container_width=True)

# FINAL MESSAGE
st.markdown("## Final Recommendation", unsafe_allow_html=True)

if decision["day_risk"] == "CRITICAL":
    st.error(decision["recommendation"])
elif decision["day_risk"] == "ELEVATED":
    st.warning(decision["recommendation"])
else:
    st.success(decision["recommendation"])
