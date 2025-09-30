import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# === 工具函数 ===
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def _to_numeric(series: pd.Series) -> pd.Series:
    """清洗金额/数量列，去掉 $ 等符号，强制转 float"""
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)  # 只保留数字和小数点
        .replace("", np.nan)  # 空值变 NaN
        .astype(float)
    )

# === 数据加载 ===
def load_transactions(db) -> pd.DataFrame:
    df = pd.DataFrame(list(db.transactions.find()))
    df = _clean_df(df)
    for col in ["Net Sales", "Gross Sales", "Qty"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    return df

def load_members(db) -> pd.DataFrame:
    df = pd.DataFrame(list(db.members.find()))
    return _clean_df(df)

def load_inventory(db) -> pd.DataFrame:
    df = pd.DataFrame(list(db.inventory.find()))
    df = _clean_df(df)
    for col in ["Qty", "Net Sales"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    return df

def load_all(time_from=None, time_to=None, db=None):
    """
    一次性加载交易 / 会员 / 库存
    兼容旧调用方式 load_all(time_from, time_to)
    也支持新方式 load_all(db=db)
    """
    if db is None:
        from services.ingestion import get_db
        db = get_db()

    tx = load_transactions(db)
    mem = load_members(db)
    inv = load_inventory(db)

    if time_from and time_to and "Datetime" in tx.columns:
        tx = tx[
            (pd.to_datetime(tx["Datetime"]) >= pd.to_datetime(time_from)) &
            (pd.to_datetime(tx["Datetime"]) <= pd.to_datetime(time_to))
        ]

    return tx, mem, inv

# === 日报表 ===
def daily_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    if "Datetime" in transactions.columns:
        transactions["date"] = pd.to_datetime(transactions["Datetime"]).dt.date
    elif "date" in transactions.columns:
        transactions["date"] = pd.to_datetime(transactions["date"]).dt.date
    else:
        return pd.DataFrame()

    summary = (
        transactions.groupby("date")
        .agg(
            net_sales=("Net Sales", "sum"),
            transactions=("Datetime", "count"),
            avg_txn=("Net Sales", "mean"),
            gross=("Gross Sales", "sum"),
            qty=("Qty", "sum"),
        )
        .reset_index()
    )
    summary["profit"] = summary["gross"] - summary["net_sales"]
    return summary

# === 销售预测 ===
def forecast_sales(transactions: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    if transactions.empty or "Datetime" not in transactions.columns:
        return pd.DataFrame()

    transactions["date"] = pd.to_datetime(transactions["Datetime"]).dt.date
    daily_sales = transactions.groupby("date")["Net Sales"].sum()

    if len(daily_sales) < 10:
        return pd.DataFrame()

    model = ExponentialSmoothing(daily_sales, trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)

    return pd.DataFrame({
        "date": pd.date_range(start=daily_sales.index[-1] + timedelta(days=1), periods=periods),
        "forecast": forecast.values
    })

# === 高消费客户 ===
def forecast_top_consumers(transactions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if transactions.empty or "Customer ID" not in transactions.columns:
        return pd.DataFrame()
    return (
        transactions.groupby("Customer ID")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )

# === SKU 消耗时序 ===
def sku_consumption_timeseries(transactions: pd.DataFrame, sku: str) -> pd.DataFrame:
    if transactions.empty or "Datetime" not in transactions.columns or "Item" not in transactions.columns:
        return pd.DataFrame()
    df = transactions[transactions["Item"] == sku].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.date
    return df.groupby("date")["Qty"].sum().reset_index()

# === 会员相关分析 ===
def member_flagged_transactions(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return transactions
    if "Customer ID" not in transactions.columns or "Customer ID" not in members.columns:
        return transactions
    member_ids = set(members["Customer ID"].unique())
    transactions = transactions.copy()
    transactions["is_member"] = transactions["Customer ID"].apply(lambda x: x in member_ids)
    return transactions

def member_frequency_stats(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    if "Customer ID" not in transactions.columns or "Datetime" not in transactions.columns:
        return pd.DataFrame()
    df = transactions.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    stats = (
        df.groupby("Customer ID")["Datetime"]
        .agg(["count", "min", "max"])
        .reset_index()
        .rename(columns={"count": "txn_count", "min": "first_txn", "max": "last_txn"})
    )
    stats["days_active"] = (stats["last_txn"] - stats["first_txn"]).dt.days.clip(lower=1)
    stats["avg_days_between"] = stats["days_active"] / stats["txn_count"]
    stats = stats[stats["Customer ID"].isin(members["Customer ID"].unique())]
    return stats

def non_member_overview(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    member_ids = set(members["Customer ID"].unique()) if not members.empty else set()
    df = transactions[~transactions["Customer ID"].isin(member_ids)].copy()
    return df.groupby("Customer ID")["Net Sales"].sum().reset_index()

# === 分类与推荐分析 ===
def category_counts(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    return transactions["Category"].value_counts().reset_index().rename(columns={"index": "Category", "Category": "count"})

def heatmap_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns or "Customer ID" not in transactions.columns:
        return pd.DataFrame()
    return pd.pivot_table(
        transactions, values="Net Sales", index="Customer ID", columns="Category", aggfunc="sum", fill_value=0
    )

def top_categories_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty or "Category" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("Category")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )

def recommend_similar_categories(transactions: pd.DataFrame, category: str, top_n: int = 3) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    other_cats = transactions["Category"].value_counts().reset_index()
    other_cats = other_cats[other_cats["index"] != category]
    return other_cats.head(top_n)

def ltv_timeseries_for_customer(transactions: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty or "Datetime" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.date
    return df.groupby("date")["Net Sales"].sum().cumsum().reset_index()

def recommend_bundles_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty or "Item" not in df.columns:
        return pd.DataFrame()
    return df["Item"].value_counts().reset_index().head(top_n)

def churn_signals_for_member(transactions: pd.DataFrame, members: pd.DataFrame, days_threshold: int = 30) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    df = transactions[transactions["Customer ID"].isin(members["Customer ID"].unique())]
    if df.empty:
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    last_seen = df.groupby("Customer ID")["Datetime"].max().reset_index()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_threshold)
    last_seen["churn_flag"] = last_seen["Datetime"] < cutoff
    return last_seen
