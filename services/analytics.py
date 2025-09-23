from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .db import get_db
from numpy.polynomial.polynomial import polyfit

# === 基础加载 ===
def _load_transactions(time_from: Optional[pd.Timestamp], time_to: Optional[pd.Timestamp]) -> pd.DataFrame:
    db = get_db()
    q = {}
    if time_from or time_to:
        q["Datetime"] = {}
        if time_from:
            q["Datetime"]["$gte"] = pd.to_datetime(time_from)
        if time_to:
            q["Datetime"]["$lt"] = pd.to_datetime(time_to + pd.Timedelta(days=1))  # 包含末日
        if not q["Datetime"]:
            q.pop("Datetime")

    rows = list(db.transactions.find(q))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    if "Qty" in df.columns:
        df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

    return df


def _load_members() -> pd.DataFrame:
    db = get_db()
    rows = list(db.members.find({}))
    return pd.DataFrame(rows)


def _load_inventory() -> pd.DataFrame:
    db = get_db()
    rows = list(db.inventory.find({}))
    return pd.DataFrame(rows)


# === 通用：给交易打会员标记 & 贴上会员信息 ===
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


# === 1) Customer Segmentation & Personalization ===
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
    if df.empty:
        return s
    s["traffic"] = len(df)
    for col in ["Product Sales", "Discounts", "Net Sales", "Gross Sales"]:
        if col in df.columns:
            s[col] = df[col].sum()
    return s


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Category" not in df.columns or "Qty" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["Category"] = df2["Category"].fillna("Unknown").astype(str)   # 🔹 修复横坐标不显示问题
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

    # ✅ 按 Category 分组，而不是 Item
    g = sub.groupby("Category")["Qty"].sum().reset_index().rename(columns={"Qty": "qty"})
    return g.sort_values("qty", ascending=False).head(topn)


def recommend_similar_categories(df: pd.DataFrame, cust_id: Optional[str] = None, topk=5) -> pd.DataFrame:
    """
    推荐热门商品类别：
    - 如果指定了 cust_id，只用该会员的交易记录计算
    - 否则用全体交易记录
    """
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
    """
    返回指定会员消费最多的类别
    """
    if df.empty or "Customer ID" not in df.columns:
        return pd.DataFrame(columns=["Category", "qty"])
    sub = df[df["Customer ID"].astype(str) == str(cust_id)]
    if sub.empty or "Category" not in sub.columns or "Qty" not in sub.columns:
        return pd.DataFrame(columns=["Category","qty"])

    g = sub.groupby("Category")["Qty"].sum().reset_index().rename(columns={"Qty":"qty"})
    return g.sort_values("qty", ascending=False).head(topn)

# === 2) Product Mix & Inventory ===
def _normalize_cols(df: pd.DataFrame) -> dict:
    """将列名规范化以便宽松匹配"""
    return {c: c.lower().strip().replace("-", " ").replace("_", " ") for c in df.columns}


def detect_store_current_qty_col(df_inv: pd.DataFrame) -> Optional[str]:
    """识别形如 'Current Quantity *' 的列"""
    if df_inv is None or df_inv.empty:
        return None
    norm = {c: c.lower().strip() for c in df_inv.columns}
    for c, n in norm.items():
        if n.startswith("current quantity"):
            return c
    return None


def restock_and_overstock_tables(
    df_inv: pd.DataFrame,
    restock_threshold: float = 3,
    clear_threshold: float = 50,
    topn: int = 15
):
    if df_inv is None or df_inv.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_inv.copy()
    qty_col = detect_store_current_qty_col(df)
    if qty_col is None:
        return pd.DataFrame(), pd.DataFrame()

    df["current_qty"] = pd.to_numeric(df[qty_col], errors="coerce")
    show_cols = [c for c in ["Item", "Item Name", "Variation Name", "current_qty"] if c in df.columns or c == "current_qty"]

    restock = df[df["current_qty"].fillna(np.inf) <= restock_threshold].copy()
    restock = restock.sort_values("current_qty").head(topn)
    restock = restock[show_cols]

    clear = df[df["current_qty"].fillna(-np.inf) > clear_threshold].copy()
    clear = clear.sort_values("current_qty", ascending=False).head(topn)
    clear = clear[show_cols]

    return restock, clear


def inventory_low_stock_alert(df_inv: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    if df_inv is None or df_inv.empty:
        return pd.DataFrame(columns=["Item", "stock_qty", "Units"])

    df = df_inv.copy()
    stock_col = detect_store_current_qty_col(df)
    if stock_col is None:
        return pd.DataFrame(columns=["Item", "stock_qty", "Units"])

    base_cols = []
    if "Item" in df.columns:
        base_cols.append("Item")
    elif "Item Name" in df.columns:
        base_cols.append("Item Name")

    out_cols = base_cols + [stock_col] + (["Units"] if "Units" in df.columns else [])
    base = df[out_cols].copy()
    base = base.rename(columns={stock_col: "stock_qty"})
    base["stock_qty"] = pd.to_numeric(base["stock_qty"], errors="coerce")
    return base[base["stock_qty"].fillna(np.inf) <= threshold].sort_values("stock_qty")


def inventory_master_table(df_inv: pd.DataFrame) -> pd.DataFrame:
    if df_inv is None or df_inv.empty:
        return pd.DataFrame()

    keep = [
        "Item", "Item Name", "Variation Name", "Unit and Precision", "SKU",
        "Reference Handle", "Stock-by Reference Handle", "Sell-by Equivalent", "Stock-by Equivalent"
    ]
    cols = [c for c in keep if c in df_inv.columns]
    qty_col = detect_store_current_qty_col(df_inv)
    if qty_col and qty_col not in cols:
        cols.append(qty_col)

    if not cols:
        return pd.DataFrame()

    out = df_inv[cols].copy()
    if qty_col and qty_col in out.columns:
        out = out.rename(columns={qty_col: "Current Quantity"})
    return out

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
    return g


def forecast_next_30days(df: pd.DataFrame) -> pd.DataFrame:
    """
    简易预测：使用最近 7 天的日均消耗（件/天）作为未来 30 天的基线。
    说明：数据点不足 7 天时退化为历史均值。
    """
    if df.empty or "date" not in df.columns or "qty" not in df.columns:
        return pd.DataFrame(columns=["date","forecast_qty"])

    df = df.sort_values("date")
    last_week = df.tail(7)
    avg_qty = last_week["qty"].mean() if not last_week.empty else df["qty"].mean()

    future_dates = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=30)
    forecast = pd.DataFrame({"date": future_dates, "forecast_qty": avg_qty})
    return forecast


def forecast_top_consumers(df_tx: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    """按 SKU 预测未来30天消耗总量：近7天日均 × 30，并给出最近7天 vs 前7天增幅作为“明显增加”判断依据"""
    if df_tx.empty or "SKU" not in df_tx.columns:
        return pd.DataFrame(columns=["SKU", "Item", "forecast_30d", "growth_ratio", "increasing"])

    df = df_tx.copy()
    df["date"] = df["Datetime"].dt.floor("D")
    g = df.groupby(["SKU", "Item", "date"])["Qty"].sum().reset_index()

    res = []
    for (sku, item), grp in g.groupby(["SKU", "Item"]):
        grp = grp.sort_values("date")
        # 最近14天拆分为 7d/7d
        last14 = grp.tail(14)
        if len(last14) >= 2:
            last7 = last14.tail(7)["Qty"].sum()
            prev7 = last14.head(len(last14) - 7)["Qty"].sum() if len(last14) >= 14 else max(grp["Qty"].sum() - last7, 1e-9)
            growth_ratio = (last7 + 1e-9) / (prev7 + 1e-9)
        else:
            growth_ratio = 1.0

        # 用最近7天平均 × 30 做简易 30 天预测
        last7_avg = grp.tail(7)["Qty"].mean() if len(grp) >= 7 else grp["Qty"].mean()
        forecast_30d = float(max(last7_avg, 0) * 30)
        res.append({"SKU": sku, "Item": item, "forecast_30d": forecast_30d, "growth_ratio": growth_ratio})

    out = pd.DataFrame(res)
    if out.empty:
        return out
    out["increasing"] = out["growth_ratio"] >= 1.3  # 显著增加阈值
    return out.sort_values("forecast_30d", ascending=False).head(topn)


def sku_consumption_timeseries(df_tx: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 SKU 或 名称关键词返回日消耗实际 + 简易预测曲线"""
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
    fc = forecast_next_30days(ds)
    return ds, fc


# === 3) Pricing & Promotion ===
def discount_breakdown(df_tx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_tx.copy()
    raw = df.get("Customer ID")
    df["is_member"] = raw.notna() & (raw.astype(str).str.strip() != "")

    used_disc = df["Discounts"] < 0 if "Discounts" in df.columns else pd.Series(False, index=df.index)
    members = df[df["is_member"] & used_disc]
    coupons = df[(~df["is_member"]) & used_disc]

    def buyers_count(x: pd.DataFrame, for_members: bool) -> int:
        if for_members and "Customer ID" in x.columns:
            return int(x["Customer ID"].nunique())
        for col in ["Transaction ID", "transaction_id", "Receipt", "Order"]:
            if col in x.columns:
                return int(x[col].astype(str).nunique())
        return int(len(x))

    def agg(x):
        return pd.Series({
            "buyers": buyers_count(x, for_members=("Customer ID" in x.columns and (x["Customer ID"].notna().any()))),
            "product_sales": x["Product Sales"].sum() if "Product Sales" in x.columns else 0,
            "discount": x["Discounts"].sum() if "Discounts" in x.columns else 0,
            "net_sales": x["Net Sales"].sum() if "Net Sales" in x.columns else 0,
            "gross_sales": x["Gross Sales"].sum() if "Gross Sales" in x.columns else 0
        })

    mem_sum, cou_sum = agg(members), agg(coupons)

    bar_df = pd.DataFrame([
        {"type": "Members (loyalty discount)", "buyers": int(mem_sum.get("buyers", 0) or 0)},
        {"type": "Non-members (coupons)", "buyers": int(cou_sum.get("buyers", 0) or 0)}
    ])

    line_df = pd.DataFrame({
        "metric": ["Product Sales", "Discounts", "Net Sales", "Gross Sales"],
        "members": [mem_sum.get("product_sales", 0), mem_sum.get("discount", 0),
                    mem_sum.get("net_sales", 0), mem_sum.get("gross_sales", 0)],
        "coupons": [cou_sum.get("product_sales", 0), cou_sum.get("discount", 0),
                    cou_sum.get("net_sales", 0), cou_sum.get("gross_sales", 0)]
    })
    return bar_df, line_df


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
    """
    改进预测：使用 Holt-Winters 指数平滑（趋势+季节性），并加上置信区间。
    """
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
        if len(y) >= 14:  # 至少两周数据才建模
            model = ExponentialSmoothing(y, trend="add", seasonal="add",
                                         seasonal_periods=7, initialization_method="estimated")
            fit = model.fit()
            forecast = fit.forecast(days)

            # 残差标准差作为置信区间近似
            resid_std = np.std(fit.resid)
            lower = forecast - 1.96 * resid_std
            upper = forecast + 1.96 * resid_std

            baseline = forecast.clip(lower=0)
            lower_ci = lower.clip(lower=0)
            upper_ci = upper.clip(lower=0)

        else:
            baseline = pd.Series([y.mean()] * days,
                                 index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))
            lower_ci = baseline * 0.9
            upper_ci = baseline * 1.1

    except Exception:
        baseline = pd.Series([y.mean()] * days,
                             index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))
        lower_ci = baseline * 0.9
        upper_ci = baseline * 1.1

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


# === 4) Retention / LTV / Churn（保持） ===
def ltv_timeseries_for_customer(df_tx: pd.DataFrame, cust_id: str, horizon_days: int = 180) -> pd.DataFrame:
    # ...（原实现）
    sub = df_tx[df_tx["Customer ID"].astype(str) == str(cust_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date","expected_ltv"])

    sub = sub.sort_values("Datetime")
    sub["prev"] = sub["Datetime"].shift()
    sub["interval"] = (sub["Datetime"] - sub["prev"]).dt.days
    delta = sub["interval"].dropna().mean()
    if not np.isfinite(delta) or delta <= 0:
        delta = 30.0

    visits = len(sub)
    total_net = sub["Net Sales"].sum() if "Net Sales" in sub.columns else 0.0
    avg_spend = total_net / max(visits, 1)

    last_visit = sub["Datetime"].max()
    r = (pd.Timestamp.today() - last_visit).days

    future = pd.date_range(pd.Timestamp.today().floor("D") + pd.Timedelta(days=1), periods=horizon_days)
    t = np.arange(1, horizon_days + 1)
    p_t = np.exp(- (r + t) / (delta + 1e-9))
    ltv_incr = p_t * avg_spend
    cum_ltv = np.cumsum(ltv_incr)

    return pd.DataFrame({"date": future, "expected_ltv": cum_ltv})


def recommend_bundles_for_customer(df_tx: pd.DataFrame, cust_id: str, topk: int = 3) -> pd.DataFrame:
    # ...（原实现）
    sub = df_tx[df_tx["Customer ID"].astype(str) == str(cust_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["strategy","details"])

    if "Category" in sub.columns:
        fav = sub.groupby("Category")["Qty"].sum().sort_values(ascending=False).head(3).index.tolist()
        if len(fav) >= 2:
            return pd.DataFrame([
                {"strategy": "Personal bundle", "details": f"Bundle {fav[0]} + {fav[1]} for {cust_id} (small % off)."}
            ])
    if "Item" in sub.columns:
        fav_items = sub.groupby("Item")["Qty"].sum().sort_values(ascending=False).head(2).index.tolist()
        if len(fav_items) >= 2:
            return pd.DataFrame([
                {"strategy": "Personal bundle", "details": f"Bundle {fav_items[0]} + {fav_items[1]} for {cust_id}."}
            ])
    return pd.DataFrame([{"strategy":"Personal bundle","details":"Bundle two most frequent choices with small discount."}])


def churn_signals_for_member(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    针对会员的潜在流失信号：
    - 最近消费间隔 vs 历史平均间隔
    - 近30天人均净销售 vs 之前平均
    * 加健壮性保护，避免 KeyError: 'days_since'
    """
    df = df_tx[df_tx["is_member"]].copy() if "is_member" in df_tx.columns else df_tx.copy()
    if df.empty or "Customer ID" not in df.columns:
        return pd.DataFrame(columns=["Customer ID","interval_ratio","spend_ratio","days_since","risk_flag"])

    df = df.sort_values(["Customer ID","Datetime"])
    df["prev"] = df.groupby("Customer ID")["Datetime"].shift()
    df["interval"] = (df["Datetime"] - df["prev"]).dt.days

    last_visit = df.groupby("Customer ID")["Datetime"].max().rename("last_visit")
    days_since = (pd.Timestamp.today() - last_visit).dt.days.rename("days_since")

    interval_avg = df.groupby("Customer ID")["interval"].mean().rename("avg_interval")
    last_interval = df.groupby("Customer ID")["interval"].last().rename("last_interval")

    df["date"] = df["Datetime"].dt.floor("D")
    recent = df[df["date"] >= (pd.Timestamp.today().floor("D") - pd.Timedelta(days=30))]
    recent_spend = recent.groupby("Customer ID")["Net Sales"].sum().rename("recent_spend") if "Net Sales" in recent.columns else pd.Series(dtype=float)
    hist_spend = df.groupby("Customer ID")["Net Sales"].mean().rename("hist_avg_spend") if "Net Sales" in df.columns else pd.Series(dtype=float)

    out = pd.concat([last_interval, interval_avg, days_since, recent_spend, hist_spend], axis=1)

    # 关键列兜底
    if "days_since" not in out.columns:
        out = out.join(days_since, how="left")
    if "avg_interval" not in out.columns:
        out = out.join(interval_avg, how="left")
    if "last_interval" not in out.columns:
        out = out.join(last_interval, how="left")
    if "recent_spend" not in out.columns:
        out["recent_spend"] = 0.0
    if "hist_avg_spend" not in out.columns:
        out["hist_avg_spend"] = out["recent_spend"].replace(0, np.nan)

    out["interval_ratio"] = out["last_interval"] / (out["avg_interval"] + 1e-9)
    out["spend_ratio"] = out["recent_spend"] / (out["hist_avg_spend"] + 1e-9)

    # 若 avg_interval 为空，则以 30 天兜底
    out["avg_interval"] = out["avg_interval"].fillna(30.0)
    out["days_since"] = out["days_since"].fillna(0)

    out["risk_flag"] = (
        (out["interval_ratio"].fillna(0) > 1.3) |
        (out["spend_ratio"].fillna(1) < 0.7) |
        (out["days_since"] > out["avg_interval"] * 1.5)
    )
    out = out.reset_index()
    return out[["Customer ID","interval_ratio","spend_ratio","days_since","risk_flag"]]


# === 5) 库存明细表（带指定列） ===

# === Facade ===
def load_all(time_from=None, time_to=None):
    tx = _load_transactions(time_from, time_to)
    mem = _load_members()
    inv = _load_inventory()
    if not tx.empty:
        tx = attach_member_info(tx, mem)
    return tx, mem, inv
