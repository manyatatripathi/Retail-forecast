# AI Retail Sales Forecasting

An interactive dashboard that uses machine learning to forecast monthly retail sales.

**Live app:** _paste your Streamlit link here_

## What it does
- Aggregates ~8,400 real orders (Superstore Sales dataset, 2009–2012) into monthly sales
- Builds features: last 1–3 months of sales, 3-month average, seasonality, quarter-end, trend
- Trains and compares a naive baseline, Linear Regression, Random Forest and Gradient Boosting
- Tests every model on the last 12 months it never saw, using WAPE (total error as % of sales)
- Forecasts 3–12 months ahead with the best model, with a likely range
- Filters by product category and region; shows feature importance and sales breakdowns

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Superstore Sales, a public sample retail dataset (Product categories: Technology, Furniture, Office Supplies; 8 Canadian regions). Customer names were removed.
