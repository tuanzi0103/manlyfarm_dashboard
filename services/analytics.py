from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .db import get_db
from pymongo.errors import PyMongoError
import streamlit as st

# === 安全查询方法，避免爆掉 ===
def _safe_find(coll, query=None, projection=None, batch_size=5000, limit=None):
    try:
        cursor = coll.find(query or {}, projection=projection, batch_size=batch_size)
        if limit:
            cursor = cursor.limit(limit)
        rows = list(cursor)
        return pd.DataFrame(rows)
    except PyMongoError as e:
        print(f"⚠️ MongoDB query failed on {coll.name}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Unexpected error on {coll.name}: {e}")
        return pd.DataFrame()

# === 基础加载 ===
def _load_transactions(time_from: Optional[pd.Timestamp], time_to: Optional[pd.Timestamp]) -> pd.DataFrame:
    db = get_db()
    q = {}
    if time_from or time_to:
        q["Datetime"] = {}
        if time_from:
            q["Datetime"]["$gte"] = pd.to_datetime(time_from)
        if time_to:
            q["Datetime"]["$lt"] = pd.to_datetime(time_to + pd.Timedelta(days=1))
        if not q["Datetime"]:
            q.pop("Datetime")

    df = _safe_find(
        db.transactions,
        query=q,
        projection={"Datetime": 1, "Qty": 1, "Net Sales": 1,
                    "Gross Sales": 1, "Discounts": 1,
                    "Customer ID": 1, "Category": 1, "location": 1}
    )
    if not df.empty and "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        if "Qty" in df.columns:
            df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    return df

def _load_members() -> pd.DataFrame:
    db = get_db()
    return _safe_find(
        db.members,
        projection={"Customer ID": 1, "First Name": 1,
                    "Surname": 1, "Email": 1, "Phone": 1}
    )

def _load_inventory() -> pd.DataFrame:
    db = get_db()
    return _safe_find(db.inventory, projection=None)

# === Member helpers ===
def attach_member_info(df_tx: pd.DataFrame, df_mem: pd.DataFrame) -> pd.DataFrame:
    df = df_tx.copy()
    cust_raw = df.get("Customer ID")
    if cust_raw is not None:
        df["is_member"] = cust_raw.notna() & (cust_raw.astype(str).str.strip() != "")
    else:
        df["is_member"] = False

    df["_cust_key"] = df.get("Customer ID").astype(str).str.strip() if "Customer ID" in df.columns else ""
    if df_mem is not None and not df_mem.empty and "Customer ID" in df_mem.columns:
        mem = df_mem.copy()
        mem["_cust_key"] = mem["Customer ID"].astype(str).str.strip()
        keep_cols = [c for c in mem.columns if c in ["_cust_key", "Customer ID", "First Name", "Surname", "Email", "Phone"]]
        df = df.merge(mem[keep_cols].drop_duplicates("_cust_key"),
                      on="_cust_key", how="left", suffixes=("", "_mem"))
        if "Customer ID" in df.columns and "Customer ID_mem" in df.columns:
            df["Customer ID"] = df["Customer ID"].fillna(df["Customer ID_mem"])
            df = df.drop(columns=["Customer ID_mem"])
    df = df.drop(columns=["_cust_key"], errors="ignore")
    return df

# === Facade (加缓存) ===
@st.cache_data(show_spinner=False)
def load_all(time_from=None, time_to=None):
    tx = _load_transactions(time_from, time_to)
    mem = _load_members()
    inv = _load_inventory()
    if not tx.empty:
        tx = attach_member_info(tx, mem)
    return tx, mem, inv

# === 1) Customers Insights ===
def member_flagged_transactions(df_tx: pd.DataFrame) -> pd.DataFrame:
    df = df_tx.copy()
    if "Customer ID" in df.columns:
        raw = df["Customer ID"]
        df["is_member"] = raw.notna() & (raw.astype(str).str.strip() != "")
    else:
        df["is_member"] = False
    return df

def member_frequency_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Customer ID" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("Customer ID").agg(
        visits=("Datetime", "count"),
        last_visit=("Datetime", "max"),
        product_sales=("Product Sales", "sum"),
        discounts=("Discounts", "sum"),
        net_sales=("Net Sales", "sum"),
        gross_sales=("Gross Sales", "sum")
    ).reset_index()
    for col in ["First Name", "Surname", "Email", "Phone"]:
        if col in df.columns:
            g[col] = df.groupby("Customer ID")[col].agg(lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan).values
    return g.sort_values("visits", ascending=False)

def non_member_overview(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(dtype=float)
    if df.empty: return s
    s["traffic"] = len(df)
    for col in ["Product Sales", "Discounts", "Net Sales", "Gross Sales"]:
        if col in df.columns: s[col] = df[col].sum()
    return s

def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Category" not in df.columns or "Qty" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["Category"] = df2["Category"].fillna("Unknown").astype(str)
    return (
        df2.groupby("Category")["Qty"].sum()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

def heatmap_pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Datetime" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["day_of_week"] = df2["Datetime"].dt.day_name()
    df2["hour"] = df2["Datetime"].dt.hour
    pv = df2.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    cats = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pv = pv.reindex(cats)
    return pv

def top_items_for_customer(df: pd.DataFrame, cust_id: str, topn=10) -> pd.DataFrame:
    if df.empty or "Customer ID" not in df.columns:
        return pd.DataFrame(columns=["Category", "qty"])
    sub = df[df["Customer ID"].astype(str) == str(cust_id)]
    if sub.empty or "Category" not in sub.columns or "Qty" not in sub.columns:
        return pd.DataFrame(columns=["Category", "qty"])
    g = sub.groupby("Category")["Qty"].sum().reset_index().rename(columns={"Qty": "qty"})
    return g.sort_values("qty", ascending=False).head(topn)

def recommend_similar_categories(df: pd.DataFrame, cust_id: Optional[str] = None, topk=5) -> pd.DataFrame:
    if df.empty or "Category" not in df.columns or "Qty" not in df.columns:
        return pd.DataFrame(columns=["Category", "count"])
    sub = df.copy()
    if cust_id is not None and "Customer ID" in sub.columns:
        sub = sub[sub["Customer ID"].astype(str) == str(cust_id)]
    if sub.empty:
        return pd.DataFrame(columns=["Category", "count"])
    sub["Category"] = sub["Category"].fillna("Unknown").astype(str)
    return (
        sub.groupby("Category")["Qty"].sum()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(topk)
    )

def top_categories_for_customer(df: pd.DataFrame, cust_id: str, topn=10) -> pd.DataFrame:
    if df.empty or "Customer ID" not in df.columns:
        return pd.DataFrame(columns=["Category", "qty"])
    sub = df[df["Customer ID"].astype(str) == str(cust_id)]
    if sub.empty or "Category" not in sub.columns or "Qty" not in sub.columns:
        return pd.DataFrame(columns=["Category","qty"])
    g = sub.groupby("Category")["Qty"].sum().reset_index().rename(columns={"Qty":"qty"})
    return g.sort_values("qty", ascending=False).head(topn)

# === 2) Sales & Forecast ===
def daily_sales(df_tx: pd.DataFrame, sku: Optional[str]=None, product_name: Optional[str]=None) -> pd.DataFrame:
    df = df_tx.copy()
    if sku and "SKU" in df.columns:
        df = df[df["SKU"].astype(str) == str(sku)]
    if product_name and "Item" in df.columns:
        df = df[df["Item"].astype(str).str.contains(product_name, case=False, na=False)]
    if df.empty:
        return pd.DataFrame(columns=["date","qty"])
    df["date"] = df["Datetime"].dt.floor("D")
    g = df.groupby("date", as_index=False)["Qty"].sum().rename(columns={"Qty": "qty"})
    g["qty"] = g["qty"].abs()
    return g

def forecast_next_30days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns or "qty" not in df.columns:
        return pd.DataFrame(columns=["date","forecast_qty"])
    df = df.sort_values("date")
    last_week = df.tail(7)
    avg_qty = last_week["qty"].mean() if not last_week.empty else df["qty"].mean()
    future_dates = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=30)
    forecast = pd.DataFrame({"date": future_dates, "forecast_qty": abs(avg_qty)})
    return forecast

def forecast_top_consumers(df_tx: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    if df_tx.empty or "SKU" not in df_tx.columns:
        return pd.DataFrame(columns=["SKU", "Item", "forecast_30d", "growth_ratio", "increasing"])
    df = df_tx.copy()
    df["date"] = df["Datetime"].dt.floor("D")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).abs()
    g = df.groupby(["SKU", "Item", "date"])["Qty"].sum().reset_index()
    res = []
    for (sku, item), grp in g.groupby(["SKU", "Item"]):
        grp = grp.sort_values("date")
        last14 = grp.tail(14)
        if len(last14) >= 2:
            last7 = last14.tail(7)["Qty"].sum()
            prev7 = last14.head(len(last14) - 7)["Qty"].sum() if len(last14) >= 14 else max(grp["Qty"].sum() - last7, 1e-9)
            growth_ratio = (last7 + 1e-9) / (prev7 + 1e-9)
        else:
            growth_ratio = 1.0
        last7_avg = grp.tail(7)["Qty"].mean() if len(grp) >= 7 else grp["Qty"].mean()
        forecast_30d = float(max(last7_avg, 0) * 30)
        res.append({"SKU": sku, "Item": item, "forecast_30d": forecast_30d, "growth_ratio": growth_ratio})
    out = pd.DataFrame(res)
    if out.empty: return out
    out["increasing"] = out["growth_ratio"] >= 1.3
    return out.sort_values("forecast_30d", ascending=False).head(topn)

def sku_consumption_timeseries(df_tx: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_tx.empty:
        return pd.DataFrame(columns=["date","qty"]), pd.DataFrame(columns=["date","forecast_qty"])
    sub = df_tx.copy()
    sub["date"] = sub["Datetime"].dt.floor("D")
    if query:
        mask = False
        if "SKU" in sub.columns:
            mask = mask | sub["SKU"].astype(str).str.contains(query, case=False, na=False)
        if "Item" in sub.columns:
            mask = mask | sub["Item"].astype(str).str.contains(query, case=False, na=False)
        sub = sub[mask]
    if sub.empty:
        return pd.DataFrame(columns=["date","qty"]), pd.DataFrame(columns=["date","forecast_qty"])
    ds = sub.groupby("date", as_index=False)["Qty"].sum().rename(columns={"Qty": "qty"})
    ds["qty"] = ds["qty"].abs()
    fc = forecast_next_30days(ds)
    return ds, fc

def ltv_timeseries_for_customer(df_tx: pd.DataFrame, cust_id: str, horizon_days: int = 180) -> pd.DataFrame:
    """
    Calculate cumulative LTV (Net Sales) for a member and project a simple forecast for horizon_days.
    """
    if df_tx.empty or "Customer ID" not in df_tx.columns or "Net Sales" not in df_tx.columns:
        return pd.DataFrame(columns=["date", "expected_ltv"])

    sub = df_tx[df_tx["Customer ID"].astype(str) == str(cust_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "expected_ltv"])

    sub["date"] = sub["Datetime"].dt.floor("D")
    daily = sub.groupby("date", as_index=False)["Net Sales"].sum().sort_values("date")
    daily["ltv"] = daily["Net Sales"].cumsum()

    # 简单线性预测（未来 horizon_days）
    if len(daily) >= 7:
        avg_daily = daily["Net Sales"].tail(7).mean()
    else:
        avg_daily = daily["Net Sales"].mean()
    future_dates = pd.date_range(start=daily["date"].max() + pd.Timedelta(days=1), periods=horizon_days)
    future_ltv = daily["ltv"].iloc[-1] + np.cumsum([avg_daily] * horizon_days)

    fut = pd.DataFrame({"date": future_dates, "expected_ltv": future_ltv})
    hist = daily[["date", "ltv"]].rename(columns={"ltv": "expected_ltv"})
    return pd.concat([hist, fut], ignore_index=True)


def recommend_bundles_for_customer(df_tx: pd.DataFrame, cust_id: str, topn: int = 3) -> pd.DataFrame:
    """
    Suggest bundle promotions for a given member based on their top categories.
    """
    if df_tx.empty or "Customer ID" not in df_tx.columns or "Category" not in df_tx.columns:
        return pd.DataFrame(columns=["strategy", "details"])

    sub = df_tx[df_tx["Customer ID"].astype(str) == str(cust_id)]
    if sub.empty:
        return pd.DataFrame(columns=["strategy", "details"])

    top_cats = (
        sub.groupby("Category")["Net Sales"].sum().reset_index().sort_values("Net Sales", ascending=False)
    )
    top_cats = top_cats["Category"].astype(str).tolist()

    suggestions = []
    if len(top_cats) >= 2:
        suggestions.append({"strategy": "Bundle Top 2", "details": f"Offer {top_cats[0]} + {top_cats[1]} at a discount."})
    if len(top_cats) >= 3:
        suggestions.append({"strategy": "Bundle Top 3", "details": f"Offer combo: {', '.join(top_cats[:3])}."})
    if len(top_cats) >= 1:
        suggestions.append({"strategy": "Upsell", "details": f"Upsell {top_cats[0]} with slow-moving items."})

    return pd.DataFrame(suggestions).head(topn)


def churn_signals_for_member(df_tx: pd.DataFrame, risk_days: int = 30) -> pd.DataFrame:
    """
    Detect churn signals: how many days since last purchase for each member, mark at-risk if > risk_days.
    """
    if df_tx.empty or "Customer ID" not in df_tx.columns or "Datetime" not in df_tx.columns:
        return pd.DataFrame(columns=["Customer ID", "days_since", "risk_flag"])

    mem = df_tx[df_tx["is_member"]] if "is_member" in df_tx.columns else df_tx.copy()
    last_purchase = mem.groupby("Customer ID")["Datetime"].max().reset_index()
    last_purchase["days_since"] = (pd.Timestamp.today().floor("D") - last_purchase["Datetime"]).dt.days
    last_purchase["risk_flag"] = last_purchase["days_since"] > risk_days

    return last_purchase[["Customer ID", "days_since", "risk_flag"]]
# === Pricing & Promotion Helpers ===
def discount_breakdown(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    按类别计算折扣金额与净销售额占比
    """
    if df_tx.empty or "Category" not in df_tx.columns or "Discounts" not in df_tx.columns:
        return pd.DataFrame(columns=["Category", "discounts", "net_sales"])
    g = df_tx.groupby("Category").agg(
        discounts=("Discounts", "sum"),
        net_sales=("Net Sales", "sum")
    ).reset_index()
    return g

def simulate_discount_forecast(df_tx: pd.DataFrame, rate: float = 0.1) -> pd.DataFrame:
    """
    简单模拟折扣策略下的收入曲线
    rate: 折扣比例（0.1 = 10%）
    """
    if df_tx.empty or "Datetime" not in df_tx.columns or "Net Sales" not in df_tx.columns:
        return pd.DataFrame(columns=["date", "simulated_revenue"])
    df = df_tx.copy()
    df["date"] = df["Datetime"].dt.floor("W")
    weekly = df.groupby("date")["Net Sales"].sum().reset_index()
    weekly["simulated_revenue"] = weekly["Net Sales"] * (1 - rate)
    return weekly

def promo_suggestions(df_tx: pd.DataFrame) -> pd.DataFrame:
    if df_tx.empty or "Category" not in df_tx.columns:
        return pd.DataFrame(columns=["strategy", "details"])

    cat_sales = df_tx.groupby("Category")["Qty"].sum().reset_index(name="qty").sort_values("qty", ascending=False)
    popular = cat_sales.head(3)["Category"].tolist()
    unpopular = cat_sales.tail(3)["Category"].tolist()

    suggestions = []
    if len(popular) >= 2:
        suggestions.append({"strategy": "Bundle popular-popular",
                            "details": f"Bundle {popular[0]} + {popular[1]} (small % off)."})
    if unpopular:
        suggestions.append({"strategy": "Bundle popular-slow",
                            "details": f"Bundle {popular[0]} + {unpopular[0]} to lift {unpopular[0]} sales."})
        if len(unpopular) > 1:
            suggestions.append({"strategy": "Discount slow movers",
                                "details": f"Lower price for {unpopular[0]} / {unpopular[1]} on weekdays."})
    return pd.DataFrame(suggestions)


def simulate_revenue_curve(df_tx: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if df_tx.empty or "Net Sales" not in df_tx.columns:
        return pd.DataFrame(columns=["date", "baseline", "lower_ci", "upper_ci",
                                     "popular_bundle", "popular_slow", "discount_slow"])

    df = df_tx.copy()
    df["date"] = df["Datetime"].dt.floor("D")
    daily_sales = df.groupby("date", as_index=False)["Net Sales"].sum().rename(columns={"Net Sales": "y"})

    if daily_sales.empty:
        return pd.DataFrame(columns=["date", "baseline", "lower_ci", "upper_ci",
                                     "popular_bundle", "popular_slow", "discount_slow"])

    daily_sales = daily_sales.sort_values("date").set_index("date").tail(180)
    y = daily_sales["y"]

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        if len(y) >= 14:
            model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=7)
            fit = model.fit()
            forecast = fit.forecast(days)

            resid_std = np.std(fit.resid)
            lower = forecast - 1.96 * resid_std
            upper = forecast + 1.96 * resid_std

            baseline = forecast.clip(lower=0)
            lower_ci = lower.clip(lower=0)
            upper_ci = upper.clip(lower=0)
        else:
            baseline = pd.Series([y.mean()] * days,
                                 index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))
            lower_ci, upper_ci = baseline * 0.9, baseline * 1.1
    except Exception:
        baseline = pd.Series([y.mean()] * days,
                             index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))
        lower_ci, upper_ci = baseline * 0.9, baseline * 1.1

    fut_df = pd.DataFrame({
        "date": baseline.index,
        "baseline": baseline.values,
        "lower_ci": lower_ci.values,
        "upper_ci": upper_ci.values
    })
    fut_df["popular_bundle"] = fut_df["baseline"] * 1.08
    fut_df["popular_slow"] = fut_df["baseline"] * 1.05
    fut_df["discount_slow"] = fut_df["baseline"] * 1.03
    return fut_df.reset_index(drop=True)
