import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def simulate_transactions(tx: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Generate synthetic daily transaction data for testing charts.
    数据覆盖过去 N 个月到今天，而不是未来。
    包含图表需要的所有字段：
    - Datetime
    - Net Sales
    - Gross Sales
    - Discounts
    - Qty
    - Category
    - location
    """
    # 确定模拟的天数
    days = months * 30

    # 结束日期设为今天
    end_date = pd.Timestamp.today().normalize()
    # 开始日期设为 N 个月前
    start_date = end_date - pd.Timedelta(days=days-1)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # 生成基准销售额
    if tx is None or tx.empty:
        base_sales = 1000
    else:
        base_sales = tx["Net Sales"].mean() if "Net Sales" in tx.columns else 1000

    rng = np.random.default_rng(seed=42)

    sales = rng.normal(loc=base_sales, scale=base_sales * 0.2, size=len(dates)).clip(min=0)
    qty = rng.poisson(lam=50, size=len(dates))
    cats = rng.choice(["retail", "bar", "other"], size=len(dates))

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
