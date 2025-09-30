import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import pandas as pd
from services.db import get_db

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
def load_transactions(db, days=365):
    """优化版: 只加载必要字段 + 限制时间范围 (默认一年)"""
    today = pd.Timestamp.today()
    start = today - pd.Timedelta(days=days)
    cursor = db.transactions.find(
        {"Datetime": {"$gte": start}},
        {"Datetime": 1, "Category": 1, "Net Sales": 1, "Gross Sales": 1, "Qty": 1}
    )
    df = pd.DataFrame(list(cursor))
    if not df.empty and "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df

def daily_summary_mongo(db, days=365, refresh=False):
    """每日汇总: 优先读 summary_daily 集合"""
    today = pd.Timestamp.today()
    start = today - pd.Timedelta(days=days)

    if not refresh:
        docs = list(db.summary_daily.find({"date": {"$gte": start.strftime("%Y-%m-%d")}}))
        if docs:
            return pd.DataFrame(docs)

    # === 重新聚合 ===
    pipeline = [
        {"$match": {"Datetime": {"$gte": start}}},
        {"$project": {
            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$Datetime"}},
            "Net Sales": 1, "Gross Sales": 1, "Qty": 1
        }},
        {"$group": {
            "_id": "$date",
            "net_sales": {"$sum": "$Net Sales"},
            "transactions": {"$sum": 1},
            "avg_txn": {"$avg": "$Net Sales"},
            "gross": {"$sum": "$Gross Sales"},
            "qty": {"$sum": "$Qty"},
        }},
        {"$project": {
            "date": "$_id",
            "net_sales": 1,
            "transactions": 1,
            "avg_txn": 1,
            "gross": 1,
            "qty": 1,
            "_id": 0
        }},
        {"$sort": {"date": 1}}
    ]
    result = list(db.transactions.aggregate(pipeline))
    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df["date"] = pd.to_datetime(df["date"])

    # === 存回 summary_daily ===
    db.summary_daily.delete_many({"date": {"$gte": start.strftime("%Y-%m-%d")}})
    db.summary_daily.insert_many(df.to_dict("records"))

    return df


def category_summary_mongo(db, days=365, refresh=False):
    """分类汇总: 优先读 summary_category 集合"""
    today = pd.Timestamp.today()
    start = today - pd.Timedelta(days=days)

    if not refresh:
        docs = list(db.summary_category.find({"date": {"$gte": start.strftime("%Y-%m-%d")}}))
        if docs:
            return pd.DataFrame(docs)

    pipeline = [
        {"$match": {"Datetime": {"$gte": start}}},
        {"$project": {
            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$Datetime"}},
            "Category": 1, "Net Sales": 1, "Gross Sales": 1, "Qty": 1
        }},
        {"$group": {
            "_id": {"date": "$date", "Category": "$Category"},
            "net_sales": {"$sum": "$Net Sales"},
            "transactions": {"$sum": 1},
            "avg_txn": {"$avg": "$Net Sales"},
            "gross": {"$sum": "$Gross Sales"},
            "qty": {"$sum": "$Qty"},
        }},
        {"$project": {
            "date": "$_id.date",
            "Category": "$_id.Category",
            "net_sales": 1,
            "transactions": 1,
            "avg_txn": 1,
            "gross": 1,
            "qty": 1,
            "_id": 0
        }},
        {"$sort": {"date": 1}}
    ]
    result = list(db.transactions.aggregate(pipeline))
    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df["date"] = pd.to_datetime(df["date"])

    # === 存回 summary_category ===
    db.summary_category.delete_many({"date": {"$gte": start.strftime("%Y-%m-%d")}})
    db.summary_category.insert_many(df.to_dict("records"))

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

def load_all(days=365, db=None):
    """从 MongoDB 加载数据（带 projection，不同集合只取关键列）"""
    if db is None:
        db = get_db()

    start = pd.Timestamp.today() - pd.Timedelta(days=days)

    # === Transaction 表需要的列（High Level, KPI, Trend） ===
    projection_tx = {
        "Datetime": 1, "Category": 1,
        "Net Sales": 1, "Gross Sales": 1,
        "Qty": 1, "Member ID": 1,
        "Discount": 1,  # Pricing & Promotion
        "_id": 0
    }

    # === Members 表需要的列（Customer Segmentation, Retention） ===
    projection_mem = {
        "First Name": 1, "Surname": 1,
        "Email": 1, "Member ID": 1,
        "Join Date": 1, "Last Purchase": 1,
        "_id": 0
    }

    # === Inventory 表需要的列（Inventory, Pricing, Velocity） ===
    projection_inv = {
        "Item Name": 1, "Item Variation Name": 1,
        "SKU": 1, "GTIN": 1,
        "Current Quantity Vie Market & Bar": 1,
        "Cost": 1, "Retail Price": 1,
        "Tax - GST (10%)": 1,
        "_id": 0
    }

    # === Mongo 查询 ===
    tx = pd.DataFrame(list(db.transactions.find(
        {"Datetime": {"$gte": start}},
        projection_tx
    )))

    mem = pd.DataFrame(list(db.members.find({}, projection_mem)))
    inv = pd.DataFrame(list(db.inventory.find({}, projection_inv)))

    # === 日期字段转换 ===
    if not tx.empty and "Datetime" in tx:
        tx["Datetime"] = pd.to_datetime(tx["Datetime"])

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
