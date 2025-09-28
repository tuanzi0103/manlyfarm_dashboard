import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def simulate_transactions(tx: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Generate synthetic daily transaction data for testing charts.
    - months: 1, 3, 6, 9
    """
    if tx is None or tx.empty:
        base_date = pd.Timestamp.today().normalize()
        base_sales = 1000
    else:
        base_date = tx["Datetime"].min().normalize()
        base_sales = tx["Net Sales"].mean() if "Net Sales" in tx.columns else 1000

    days = months * 30
    dates = pd.date_range(base_date, periods=days, freq="D")

    rng = np.random.default_rng(seed=42)
    sales = rng.normal(loc=base_sales, scale=base_sales * 0.2, size=days).clip(min=0)
    qty = rng.poisson(lam=50, size=days)
    cats = rng.choice(["retail", "bar", "other"], size=days)

    df = pd.DataFrame({
        "Datetime": dates,
        "Net Sales": sales,
        "Gross Sales": sales * 1.2,
        "Discounts": -sales * 0.05,
        "Qty": qty,
        "Category": cats,
        "location": cats,  # 🔹 保证和真实数据格式一致
    })

    return df
