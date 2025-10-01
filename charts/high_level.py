import streamlit as st
import pandas as pd
import plotly.express as px
from services.db import get_db

def _safe_sum(df, col):
    """对任意列安全求和：先清洗成数值，再 sum，避免 str+int 报错"""
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = df[col]

    # 已经是数值列
    if pd.api.types.is_numeric_dtype(s):
        return float(pd.to_numeric(s, errors="coerce").sum(skipna=True))

    # 可能是 "$1,234"、"1,234" 之类的字符串
    s = (
        s.astype(str)
         .str.replace(r"[^0-9\.\-]", "", regex=True)  # 去掉 $, 逗号等
         .replace("", pd.NA)
    )
    return float(pd.to_numeric(s, errors="coerce").sum(skipna=True) or 0.0)

def persisting_multiselect(label, options, key, default=None):
    """多选框 + 记忆功能"""
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options, default=st.session_state[key], key=key)

# === 改成 SQLite 查询 ===
@st.cache_data
def get_high_level_data(days=365):
    db = get_db()

    daily = pd.read_sql(
        """
        SELECT 
            date(Datetime) as date,
            SUM([Net Sales]) as net_sales,
            COUNT(*) as transactions,
            AVG([Net Sales]) as avg_txn,
            SUM([Gross Sales]) as gross,
            SUM(Qty) as qty
        FROM transactions
        GROUP BY date
        ORDER BY date
        """,
        db
    )

    category = pd.read_sql(
        """
        SELECT 
            date(Datetime) as date,
            Category,
            SUM([Net Sales]) as net_sales,
            COUNT(*) as transactions,
            AVG([Net Sales]) as avg_txn,
            SUM([Gross Sales]) as gross,
            SUM(Qty) as qty
        FROM transactions
        GROUP BY date, Category
        ORDER BY date
        """,
        db
    )

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])

    return daily, category

def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    daily, category = get_high_level_data()

    if daily.empty or category.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    today = pd.Timestamp.today().normalize()
    latest_date = daily["date"].max()
    df_latest = daily[daily["date"] == latest_date]

    # === KPI ===
    kpis = {
        "Daily Net Sales": df_latest["net_sales"].sum(),
        "Daily Transactions": df_latest["transactions"].sum(),
        "Avg Transaction": df_latest["avg_txn"].mean(),
        "3M Avg": daily["net_sales"].rolling(90, min_periods=1).mean().iloc[-1],
        "6M Avg": daily["net_sales"].rolling(180, min_periods=1).mean().iloc[-1],
        "Inventory Value": _safe_sum(inv, "Current Quantity Vie Market & Bar"),
        "Profit (Amount)": df_latest["gross"].sum() - df_latest["net_sales"].sum(),
        "Items Sold": df_latest["qty"].sum(),
    }

    st.markdown(f"### 📅 Latest available date: {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")

    labels, values = list(kpis.keys()), list(kpis.values())
    for row in range(0, len(labels), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row + i
            if idx < len(labels):
                display = "-" if pd.isna(values[idx]) else f"{values[idx]:,.2f}"
                with col:
                    st.markdown(f"<div style='font-size:28px; font-weight:600'>{display}</div>", unsafe_allow_html=True)
                    st.caption(labels[idx])

    st.markdown("---")

    # === 交互选择 ===
    st.subheader("🔍 Select Parameters")
    time_range = persisting_multiselect("Choose time range", ["Custom dates", "WTD", "MTD", "YTD"], key="hl_time")
    data_options = ["Daily Net Sales","Daily Transactions","Avg Transaction","3M Avg","6M Avg","Inventory Value","Profit (Amount)","Items Sold"]
    data_sel = persisting_multiselect("Choose data type", data_options, key="hl_data")

    # —— 改动 1：这里新增大类 & 选项扩展 ——  #
    bar_cats = {"Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"}
    all_cats = sorted(category["Category"].fillna("Unknown").unique().tolist())
    all_cats_extended = all_cats + ["bar", "retail"]
    cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    if time_range and data_sel and cats_sel:
        grouped = category.copy()

        # ✅ 时间过滤
        if "WTD" in time_range:
            grouped = grouped[grouped["date"] >= today - pd.Timedelta(days=7)]
        if "MTD" in time_range:
            grouped = grouped[grouped["date"] >= today - pd.Timedelta(days=30)]
        if "YTD" in time_range:
            grouped = grouped[grouped["date"] >= today - pd.Timedelta(days=365)]
        if "Custom dates" in time_range:
            t1 = st.date_input("From")
            t2 = st.date_input("To")
            if t1 and t2:
                grouped = grouped[(grouped["date"] >= pd.to_datetime(t1)) & (grouped["date"] <= pd.to_datetime(t2))]

        # —— 改动 2：这里替换“类别过滤”逻辑，支持 大类+小类 共存 ——  #
        small_cats = [c for c in cats_sel if c not in ("bar", "retail")]
        parts = []

        # 小类：保持原始明细（不聚合）
        if small_cats:
            parts.append(grouped[grouped["Category"].isin(small_cats)])

        # 大类：bar（5 个子类的汇总）
        if "bar" in cats_sel:
            bar_df = grouped[grouped["Category"].isin(list(bar_cats))]
            if not bar_df.empty:
                agg = (bar_df.groupby("date", as_index=False)
                       .agg(net_sales=("net_sales","sum"),
                            transactions=("transactions","sum"),
                            gross=("gross","sum"),
                            qty=("qty","sum")))
                # 重新计算汇总后的 avg_txn = net_sales / transactions
                agg["avg_txn"] = (agg["net_sales"] / agg["transactions"]).replace([pd.NA, float("inf")], 0)
                agg["Category"] = "bar"
                parts.append(agg)

        # 大类：retail（非 bar 的全部汇总）
        if "retail" in cats_sel:
            retail_df = grouped[~grouped["Category"].isin(list(bar_cats))]
            if not retail_df.empty:
                agg = (retail_df.groupby("date", as_index=False)
                       .agg(net_sales=("net_sales","sum"),
                            transactions=("transactions","sum"),
                            gross=("gross","sum"),
                            qty=("qty","sum")))
                agg["avg_txn"] = (agg["net_sales"] / agg["transactions"]).replace([pd.NA, float("inf")], 0)
                agg["Category"] = "retail"
                parts.append(agg)

        # 合并
        if parts:
            grouped = pd.concat(parts, ignore_index=True)
        else:
            grouped = grouped.iloc[0:0]

        mapping = {
            "Daily Net Sales": ("net_sales", "Daily Net Sales"),
            "Daily Transactions": ("transactions", "Daily Transactions"),
            "Avg Transaction": ("avg_txn", "Avg Transaction"),
            "3M Avg": ("net_sales", "3M Avg (Rolling 90d)"),
            "6M Avg": ("net_sales", "6M Avg (Rolling 180d)"),
            "Inventory Value": ("net_sales", "Inventory Value (static)"),
            "Profit (Amount)": ("gross", "Profit (Gross-Net)"),
            "Items Sold": ("qty", "Items Sold"),
        }

        # === 图表 ===
        for metric in data_sel:
            if metric not in mapping:
                continue
            y, colname = mapping[metric]

            plot_df = grouped.dropna(subset=[y])
            if metric in ["3M Avg", "6M Avg"]:
                if metric == "3M Avg":
                    plot_df["rolling"] = plot_df.groupby("Category")[y].transform(lambda x: x.rolling(90, min_periods=1).mean())
                else:
                    plot_df["rolling"] = plot_df.groupby("Category")[y].transform(lambda x: x.rolling(180, min_periods=1).mean())
                fig = px.line(plot_df, x="date", y="rolling", color="Category", title=colname, markers=True)
            elif metric == "Profit (Amount)":
                plot_df["profit"] = plot_df["gross"] - plot_df["net_sales"]
                fig = px.line(plot_df, x="date", y="profit", color="Category", title=colname, markers=True)
            else:
                fig = px.line(plot_df, x="date", y=y, color="Category", title=colname, markers=True)

            fig.update_layout(xaxis=dict(type="date"))
            st.plotly_chart(fig, use_container_width=True)

        # === 表格 ===
        st.subheader("📋 Detailed Data")
        cols_to_show = ["date", "Category"]
        for sel in data_sel:
            if sel in mapping:
                cols_to_show.append(mapping[sel][0])

        st.dataframe(grouped[cols_to_show].assign(date=grouped["date"].dt.strftime("%Y-%m-%d")), use_container_width=True)

    else:
        st.info("Please select time range, data, and category to generate the chart.")
