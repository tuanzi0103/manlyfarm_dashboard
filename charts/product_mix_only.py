import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import re
from itertools import combinations

from services.db import get_db
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==================== 可调参数 ====================
FORECAST_WEEKS = 3  # 预测3周
MIN_SERIES_LEN_FOR_HOLT = 10
RECENT_DAYS_FOR_VELOCITY = 30

# ==================== 小工具 ====================
def _persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)

def _item_col(df: pd.DataFrame) -> str:
    for c in ["Item", "Item Name", "Variation Name"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _category_col(df: pd.DataFrame) -> str | None:
    for c in ["Category", "Categories", "Category Name", "Category (Top Level)",
              "Reporting Category", "Department"]:
        if c in df.columns:
            return c
    return None

def _order_key_cols(df: pd.DataFrame):
    for col in ["Order ID", "Receipt ID", "Txn ID", "Transaction ID"]:
        if col in df.columns:
            return col
    return None

def _format_dmy(x):
    if pd.isna(x) or x is None or str(x).strip() in ["", "0"]:
        return ""
    d = pd.to_datetime(x, errors="coerce")
    return "" if pd.isna(d) else d.strftime("%d/%m/%Y")

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# ==================== 组合建议 ====================
def _build_category_pairs_by_order(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    df2 = df.copy()
    order_col = _order_key_cols(df2)
    if order_col is not None:
        order_key = df2[order_col].astype(str)
    else:
        if "Datetime" in df2.columns:
            key = pd.to_datetime(df2["Datetime"], errors="coerce").dt.floor("min").astype(str)
            if "Customer ID" in df2.columns:
                key = key + "_" + df2["Customer ID"].astype(str)
            order_key = key
        else:
            order_key = pd.Series(["__one__"] * len(df2), index=df2.index)
    df2 = df2[[cat_col]].assign(order_key=order_key)

    pair_counter = {}
    for _, g in df2.groupby("order_key"):
        cats = sorted(set(g[cat_col].dropna().astype(str).str.strip()))
        cats = [c for c in cats if c and c.lower() != "none"]
        if len(cats) < 2:
            continue
        for a, b in combinations(cats, 2):
            key = (a, b)
            pair_counter[key] = pair_counter.get(key, 0) + 1

    if not pair_counter:
        return pd.DataFrame(columns=["a", "b", "count"])

    return (pd.DataFrame([(k[0], k[1], v) for k, v in pair_counter.items()],
                         columns=["a", "b", "count"])
            .sort_values("count", ascending=False))

def _strategy_suggestions_by_category(df: pd.DataFrame, top_pairs: int = 6, slow_k: int = 6) -> dict:
    if df.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    cat_col = _category_col(df)
    if cat_col is None or "Qty" not in df.columns:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    df = df[df[cat_col].notna()]
    df = df[df[cat_col].astype(str).str.strip().str.lower() != "none"]
    if df.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    cat_cnt = (df.groupby(cat_col)["Qty"]
               .sum()
               .reset_index(name="qty")
               .sort_values("qty", ascending=False))

    if cat_cnt.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    q75, q25 = cat_cnt["qty"].quantile(0.75), cat_cnt["qty"].quantile(0.25)
    popular = cat_cnt[cat_cnt["qty"] >= q75][cat_col].astype(str).tolist()
    slow    = cat_cnt[cat_cnt["qty"] <= q25][cat_col].astype(str).tolist()

    pairs = _build_category_pairs_by_order(df, cat_col)

    pp, ps = [], []
    if not pairs.empty and popular:
        mask = pairs["a"].isin(popular) & pairs["b"].isin(popular)
        pp = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())

    if not pairs.empty and popular and slow:
        mask = ((pairs["a"].isin(popular) & pairs["b"].isin(slow)) |
                (pairs["a"].isin(slow) & pairs["b"].isin(popular)))
        ps = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())

    if not ps and popular and slow:
        for a, b in zip(popular[:top_pairs], slow[:top_pairs]):
            ps.append(f"{a} + {b}")

    return {"popular_popular": pp, "popular_slow": ps, "discount_slow": slow[:slow_k]}

# ==================== 历史序列 & 预测 ====================
def _weekly_category_revenue(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    if "Net Sales" not in df.columns or "Datetime" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    cat_col = _category_col(df)
    if cat_col is None:
        return pd.DataFrame(columns=["date", "value"])

    clean = [c.strip() for c in categories if str(c).strip() and str(c).strip().lower() != "none"]
    sub = df[df[cat_col].astype(str).isin(clean)]
    if sub.empty:
        return pd.DataFrame(columns=["date", "value"])
    sub = sub.copy()
    sub["date"] = pd.to_datetime(sub["Datetime"], errors="coerce").dt.to_period("W").dt.start_time
    return (sub.groupby("date")["Net Sales"].sum().reset_index(name="value")
            .sort_values("date"))

from sklearn.metrics import mean_absolute_percentage_error

def _holt_weekly_forecast(series_df: pd.DataFrame, forecast_weeks: int = FORECAST_WEEKS):
    if series_df.empty:
        return pd.DataFrame(columns=["date", "yhat"]), 0.0

    s = series_df.set_index(pd.to_datetime(series_df["date"]))["value"].asfreq("W").fillna(0.0)

    if s.size < MIN_SERIES_LEN_FOR_HOLT:
        # ⚡ 改为最近4周均线
        const = float(s.tail(4).mean()) if not s.empty else 0.0
        fut_idx = pd.date_range(start=(s.index.max() if len(s) else pd.Timestamp.today()) + pd.Timedelta(weeks=1),
                                periods=forecast_weeks, freq="W")
        return pd.DataFrame({"date": fut_idx.date, "yhat": [const]*forecast_weeks}), 0.0

    try:
        # ⚡ 用月度季节性（4 周一个周期）
        model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=4, damped_trend=True)
        fit = model.fit(optimized=True, use_brute=True)

        # ---- 计算预测准确率 ----
        split_point = int(len(s) * 0.8)
        train, test = s.iloc[:split_point], s.iloc[split_point:]
        if len(test) > 0:
            model_val = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=4, damped_trend=True).fit()
            pred = model_val.forecast(len(test))
            mape = 1 - mean_absolute_percentage_error(test, pred)
        else:
            mape = 0.0

        fut = fit.forecast(forecast_weeks)
        return pd.DataFrame({"date": fut.index.date, "yhat": fut.values}), float(round(mape, 3))

    except Exception:
        try:
            # fallback → Holt 线性趋势
            model = ExponentialSmoothing(s, trend="add")
            fit = model.fit()
            fut = fit.forecast(forecast_weeks)
            return pd.DataFrame({"date": fut.index.date, "yhat": fut.values}), 0.0
        except Exception:
            const = float(s.tail(4).mean()) if not s.empty else 0.0
            fut_idx = pd.date_range(start=s.index.max()+pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W")
            return pd.DataFrame({"date": fut_idx.date, "yhat": [const]*forecast_weeks}), 0.0

# ==================== 缓存 ====================
@st.cache_data(show_spinner=False, persist=True)
def _precompute(tx):
    tx = tx.copy()
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx["date"] = tx["Datetime"].dt.date
    return tx

@st.cache_data(show_spinner=False, persist=True)
def load_inventory():
    conn = get_db()
    try:
        return pd.read_sql("SELECT * FROM inventory", conn)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, persist=True)
def compute_combo_forecast_category(tx, combo):
    cats = [s.strip() for s in str(combo).split("+") if s and s.strip()]
    series = _weekly_category_revenue(tx, cats)
    return _holt_weekly_forecast(series, forecast_weeks=FORECAST_WEEKS)

# ==================== 页面入口 ====================
def show_product_mix_only(tx: pd.DataFrame):
    st.header("📊 Product Mix")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    # 🔹 统一 Datetime/date 类型为 Timestamp
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx["date"] = tx["Datetime"].dt.normalize()   # 保持 Timestamp，不转成 date

    # --------- 建议 ---------
    st.subheader("💡 Discount Forecast Suggestions")
    cat_col = _category_col(tx)
    item_col = _item_col(tx)
    base_cols = [c for c in [cat_col or "", item_col, "Qty", "Datetime", "Customer ID"] if c and c in tx.columns]

    input_df = tx[base_cols].copy()
    if cat_col:
        mask_valid_cat = input_df[cat_col].notna() & (input_df[cat_col].astype(str).str.strip().str.lower() != "none")
        input_df = input_df[mask_valid_cat]

    sugg = _strategy_suggestions_by_category(input_df, top_pairs=6, slow_k=6)
    rows = []
    for p in sugg.get("popular_popular", []): rows.append({"strategy": "Bundle popular-popular", "combo": p})
    for p in sugg.get("popular_slow", []):    rows.append({"strategy": "Bundle popular-slow", "combo": p})
    for s in sugg.get("discount_slow", []):   rows.append({"strategy": "Discount slow mover", "combo": s})
    sugg_df = pd.DataFrame(rows, columns=["strategy", "combo"])
    if not sugg_df.empty:
        st.table(sugg_df)
    else:
        st.info("No suggestions available based on current data.")

    # --------- Inventory KPIs ---------
    st.subheader("📦 Inventory Details")

    conn = get_db()
    try:
        inv = pd.read_sql("SELECT * FROM inventory", conn)
    except Exception:
        inv = pd.DataFrame()

    if inv.empty:
        st.info("Inventory table not available.")
        return

    inv_key = "Item Name" if "Item Name" in inv.columns else None
    tx_key  = "Item" if "Item" in tx.columns else None
    if not inv_key or not tx_key:
        st.info("Cannot align items between inventory and transactions.")
        return

    inv[inv_key] = _norm(inv[inv_key])
    tx[tx_key]   = _norm(tx[tx_key])

    sold = tx.groupby(tx_key, as_index=False)["Qty"].sum().rename(columns={"Qty": "Sold"})
    last_sold = tx.dropna(subset=["Datetime"]).groupby(tx_key, as_index=False)["Datetime"].max().rename(columns={"Datetime": "Last sold"})

    df = inv.copy()
    out = pd.DataFrame({
        "Item Name": df[inv_key],
        "Variation Name": df.get("Variation Name", ""),
        "GTIN": df.get("GTIN", ""),
        "SKU": df.get("SKU", ""),
    })

    if "Current Quantity Vie Market & Bar" in inv.columns:
        onhand = pd.to_numeric(inv["Current Quantity Vie Market & Bar"], errors="coerce")
    elif "Stock on Hand" in inv.columns:
        onhand = pd.to_numeric(inv["Stock on Hand"], errors="coerce")
    else:
        onhand = np.nan
    out["On hand"] = onhand

    out = out.merge(sold, left_on="Item Name", right_on=tx_key, how="left")
    out = out.merge(last_sold, left_on="Item Name", right_on=tx_key, how="left")

    # === Sell-through 百分比 ===
    sold_num = pd.to_numeric(out.get("Sold", np.nan), errors="coerce").fillna(0)
    onh_num  = pd.to_numeric(out.get("On hand", np.nan), errors="coerce").fillna(0)
    ratio = sold_num / np.maximum(1e-9, sold_num + onh_num)
    out["Sell-through"] = (ratio * 100).round(0).astype(int).astype(str) + "%"

    # === Velocity & Out of stock ===
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=30)

    # 过去30天销量 → 日均销量（保证都是 Timestamp）
    daily_sales = (
        tx[tx["date"] >= cutoff]
        .groupby(tx_key)["Qty"].sum() / 30
    ).to_dict()

    def calc_velocity(item):
        v_day = daily_sales.get(item, 0)
        return int(round(v_day * 30))  # 转 per month

    def calc_out_of_stock(row):
        onhand = pd.to_numeric(row["On hand"], errors="coerce")
        v_day = daily_sales.get(row["Item Name"], 0)
        if pd.isna(onhand) or onhand <= 0:
            return today.strftime("%d/%m/%Y")
        if v_day > 0:
            days_left = int(onhand / v_day)
            return (today + pd.Timedelta(days=days_left)).strftime("%d/%m/%Y")
        return ""

    out["Sales velocity"] = out["Item Name"].apply(calc_velocity).astype(str) + " per month"
    out["Out of stock"] = out.apply(calc_out_of_stock, axis=1)

    # Last received（智能识别）
    lr_col = None
    for c in inv.columns:
        cl = c.lower().strip()
        if cl in {"last received", "last_received"} or re.search(r"last.*receiv", cl) or re.search(r"(received.*date|date.*received)", cl):
            lr_col = c
            break
    out["Last received"] = inv[lr_col] if lr_col else ""

    # 日期格式化
    for dcol in ["Last sold", "Last received"]:
        if dcol in out.columns:
            out[dcol] = out[dcol].apply(_format_dmy)

    show_cols = ["Item Name", "Variation Name", "GTIN", "SKU",
                 "Sell-through", "On hand", "Sold", "Sales velocity",
                 "Last sold", "Last received", "Out of stock"]
    st.dataframe(out[show_cols], use_container_width=True)
