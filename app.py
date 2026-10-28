"""AI Retail Sales Forecasting — Streamlit dashboard.

Data: Superstore Sales (a public retail dataset, 2009–2012, ~8,400 orders).
Models: Naive baseline, Linear Regression, Random Forest, Gradient Boosting.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def wape(actual, pred) -> float:
    """Weighted absolute percentage error: total error as a % of total sales."""
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum() * 100)

st.set_page_config(page_title="AI Retail Sales Forecasting", page_icon="🛒", layout="wide")

ACTUAL = "#1f6f8b"
FORECAST = "#e07a1f"
MUTED = "#b8c4cc"
TEST_MONTHS = 12


# ---------- Data ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/superstore_sales.csv", parse_dates=["order_date"])
    return df


def monthly_series(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("order_date").resample("MS")["sales"].sum()
    return s.asfreq("MS", fill_value=0.0)


# ---------- Features ----------
def make_features(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    X["lag_1"] = s.shift(1)
    X["lag_2"] = s.shift(2)
    X["lag_3"] = s.shift(3)
    X["rolling_mean_3"] = s.shift(1).rolling(3).mean()
    X["month_sin"] = np.sin(2 * np.pi * s.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * s.index.month / 12)
    X["quarter_end"] = s.index.month.isin([3, 6, 9, 12]).astype(int)
    X["trend"] = np.arange(len(s))
    return X


FEATURE_NAMES = {
    "lag_1": "Sales last month",
    "lag_2": "Sales 2 months ago",
    "lag_3": "Sales 3 months ago",
    "rolling_mean_3": "3-month average",
    "month_sin": "Season (month, sin)",
    "month_cos": "Season (month, cos)",
    "quarter_end": "Quarter-end month",
    "trend": "Long-term trend",
}


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    }


def recursive_forecast(model, history: pd.Series, steps: int) -> pd.Series:
    """Predict one month at a time, feeding each prediction back in as history."""
    hist = history.copy()
    for _ in range(steps):
        nxt = hist.index[-1] + pd.offsets.MonthBegin(1)
        hist.loc[nxt] = np.nan
        X = make_features(hist).iloc[[-1]]
        hist.loc[nxt] = max(float(model.predict(X)[0]), 0.0)
    return hist.iloc[-steps:]


@st.cache_data
def evaluate(s: pd.Series):
    """Train on all but the last 12 months, test on those 12 months."""
    train, test = s.iloc[:-TEST_MONTHS], s.iloc[-TEST_MONTHS:]
    Xtr = make_features(train).dropna()
    ytr = train.loc[Xtr.index]

    results, test_preds = [], {}
    # Naive baseline: same month last year
    naive = s.shift(12).iloc[-TEST_MONTHS:]
    results.append(("Naive (same month last year)", wape(test, naive),
                    mean_absolute_error(test, naive)))
    test_preds["Naive (same month last year)"] = naive

    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        pred = recursive_forecast(model, train, TEST_MONTHS)
        pred.index = test.index
        results.append((name, wape(test, pred), mean_absolute_error(test, pred)))
        test_preds[name] = pred

    table = pd.DataFrame(results, columns=["Model", "Error (%)", "MAE ($)"]).sort_values("Error (%)")
    return table.reset_index(drop=True), test_preds


@st.cache_data
def fit_and_forecast(s: pd.Series, model_name: str, horizon: int, error_pct: float):
    X = make_features(s).dropna()
    model = build_models()[model_name]
    model.fit(X, s.loc[X.index])
    fc = recursive_forecast(model, s, horizon)
    band = min(error_pct, 90) / 100
    lower, upper = fc * (1 - band), fc * (1 + band)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X.columns)
    return fc, lower, upper, importance


# ---------- UI ----------
df = load_data()

st.title("AI retail sales forecasting")
st.caption("Superstore Sales dataset · machine-learning forecast of monthly revenue")

with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Product category", ["All"] + sorted(df["category"].unique()))
    region = st.selectbox("Region", ["All"] + sorted(df["region"].unique()))
    horizon = st.slider("Months to forecast", 3, 12, 6)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Orders are summed into monthly sales.\n"
        "2. Features are built: recent sales, 3-month average, season, trend.\n"
        "3. Four models are trained on the first 3 years and tested on the last 12 months.\n"
        "4. The most accurate model forecasts the months ahead."
    )

view = df.copy()
if category != "All":
    view = view[view["category"] == category]
if region != "All":
    view = view[view["region"] == region]

s = monthly_series(view)
table, test_preds = evaluate(s)
ml_table = table[~table["Model"].str.startswith("Naive")]
best = ml_table.iloc[0]
fc, lower, upper, importance = fit_and_forecast(s, best["Model"], horizon, float(best["Error (%)"]))

last_12 = s.iloc[-12:].sum()
fc_total = fc.sum()
same_period_last_year = s.iloc[-12:-12 + horizon].sum() if horizon < 12 else s.iloc[-12:].sum()
change = (fc_total / same_period_last_year - 1) * 100 if same_period_last_year else 0
naive_err = float(table.loc[table["Model"].str.startswith("Naive"), "Error (%)"].iloc[0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sales, last 12 months", f"${last_12:,.0f}")
c2.metric(f"Forecast, next {horizon} months", f"${fc_total:,.0f}", f"{change:+.1f}% vs same months last year")
c3.metric("Best model", best["Model"])
c4.metric("Forecast accuracy", f"{max(0.0, 100 - best['Error (%)']):.1f}%",
          f"{naive_err - best['Error (%)']:+.1f} pts vs naive baseline")

if best['Error (%)'] > 50:
    st.warning("This selection has too few orders per month for a reliable forecast. "
               "Try 'All' regions or a whole category.")

# Main chart
st.subheader("Monthly sales: actual, test and forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Actual sales", line=dict(color=ACTUAL, width=2)))
tp = test_preds[best["Model"]]
fig.add_trace(go.Scatter(x=tp.index, y=tp.values, name=f"{best['Model']} on test months",
                         line=dict(color=FORECAST, width=2, dash="dot")))
fig.add_trace(go.Scatter(x=list(upper.index) + list(lower.index[::-1]),
                         y=list(upper.values) + list(lower.values[::-1]),
                         fill="toself", fillcolor="rgba(224,122,31,0.15)", line=dict(width=0),
                         name="Likely range", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[s.index[-1]] + list(fc.index), y=[s.iloc[-1]] + list(fc.values),
                         name="Forecast", line=dict(color=FORECAST, width=3)))
fig.add_vline(x=s.index[-TEST_MONTHS], line_dash="dash", line_color=MUTED)
fig.add_annotation(x=s.index[-TEST_MONTHS], y=1, yref="paper", text="Test period starts",
                   showarrow=False, xanchor="left", font=dict(color="gray"))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                  yaxis_title="Sales ($)", legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Model comparison")
    st.caption("Forecast error on the 12 held-out months (WAPE: total error as a % of total sales). Lower is better.")
    colors = [FORECAST if m == best["Model"] else MUTED for m in table["Model"]]
    bar = go.Figure(go.Bar(x=table["Error (%)"], y=table["Model"], orientation="h", marker_color=colors,
                           text=table["Error (%)"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
    bar.update_layout(height=300, margin=dict(l=10, r=40, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="Error (%)")
    st.plotly_chart(bar, width="stretch")

with right:
    st.subheader("What drives the forecast")
    if importance is not None:
        imp = importance.rename(index=FEATURE_NAMES).sort_values()
        fi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACTUAL))
        fi.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Importance")
        st.caption(f"Feature importance from the {best['Model']} model.")
        st.plotly_chart(fi, width="stretch")
    else:
        st.info("Linear Regression won for this selection, so there is no tree-based feature importance to show.")

st.subheader("Where the sales come from")
a, b = st.columns(2)
with a:
    by_cat = view.groupby("category", as_index=False)["sales"].sum().sort_values("sales")
    st.plotly_chart(px.bar(by_cat, x="sales", y="category", orientation="h",
                           labels={"sales": "Sales ($)", "category": ""}, color_discrete_sequence=[ACTUAL])
                    .update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
with b:
    by_reg = view.groupby("region", as_index=False)["sales"].sum().sort_values("sales")
    st.plotly_chart(px.bar(by_reg, x="sales", y="region", orientation="h",
                           labels={"sales": "Sales ($)", "region": ""}, color_discrete_sequence=[ACTUAL])
                    .update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")

with st.expander("Forecast table"):
    out = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast ($)": fc.round(0),
                        "Low ($)": lower.round(0), "High ($)": upper.round(0)})
    st.dataframe(out, hide_index=True, width="stretch")
    st.download_button("Download forecast as CSV", out.to_csv(index=False), "forecast.csv", "text/csv")

with st.expander("Model results table"):
    st.dataframe(table.round(2), hide_index=True, width="stretch")

st.caption(f"Data: Superstore Sales, {len(df):,} orders, {df['order_date'].min():%b %Y} – "
           f"{df['order_date'].max():%b %Y}. Customer names removed.")
