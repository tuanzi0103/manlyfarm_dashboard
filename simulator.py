import pandas as pd
import numpy as np

def simulate_transactions(tx: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Generate synthetic daily transaction data for testing charts.
    - months: 1, 3, 6, 9
    """
    if tx.empty:
        # 如果原始数据为空，生成一个基线
        base_date = pd.Timestamp.today().normalize()
        base_sales = 1000
    else:
        base_date = tx["Datetime"].min().normalize()
        base_sales = tx["Net Sales"].mean() if "Net Sales" in tx.columns else 1000

    days = months * 30
    dates = pd.date_range(base_date, periods=days, freq="D")

    rng = np.random.default_rng(seed=42)  # 固定种子可复现
    sales = rng.normal(loc=base_sales, scale=base_sales * 0.2, size=days).clip(min=0)
    qty = rng.poisson(lam=50, size=days)

    df = pd.DataFrame({
        "Datetime": dates,
        "Net Sales": sales,
        "Gross Sales": sales * 1.2,  # 假设毛利率 20%
        "Discounts": -sales * 0.05,  # 假设平均折扣 5%
        "Qty": qty,
        "Category": rng.choice(["Retail", "Bar", "Other"], size=days)
    })

    return df
