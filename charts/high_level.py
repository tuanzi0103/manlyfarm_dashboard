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
@st.cache_data(persist="disk")
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

    # === Category 选择 ===
    bar_cats = ["Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"]
    all_cats = sorted(category["Category"].fillna("Unknown").unique().tolist())
    all_cats_extended = all_cats + ["bar", "retail"]  # ✅ 新增 bar/retail

    cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    if time_range and data_sel and cats_sel:
        grouped = category.copy()

        # 构建超级类别
        grouped["super_cat"] = grouped["Category"].fillna("Unknown")
        grouped.loc[grouped["Category"].isin(bar_cats), "super_cat"] = "bar"
        grouped.loc[~grouped["Category"].isin(bar_cats), "super_cat"] = "retail"

        # 用户选择过滤
        if any(x in ["bar", "retail"] for x in cats_sel):
            grouped = grouped.groupby(["date", "super_cat"], as_index=False).agg({
                "net_sales": "sum",
                "transactions": "sum",
                "avg_txn": "mean",
                "gross": "sum",
                "qty": "sum"
            })
            grouped = grouped[grouped["super_cat"].isin(cats_sel)]
            cat_field = "super_cat"
        else:
            grouped = grouped[grouped["Category"].isin(cats_sel)]
            cat_field = "Category"

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
                    plot_df["rolling"] = plot_df.groupby(cat_field)[y].transform(lambda x: x.rolling(90, min_periods=1).mean())
                else:
                    plot_df["rolling"] = plot_df.groupby(cat_field)[y].transform(lambda x: x.rolling(180, min_periods=1).mean())
                fig = px.line(plot_df, x="date", y="rolling", color=cat_field, title=colname, markers=True)
            elif metric == "Profit (Amount)":
                plot_df["profit"] = plot_df["gross"] - plot_df["net_sales"]
                fig = px.line(plot_df, x="date", y="profit", color=cat_field, title=colname, markers=True)
            else:
                fig = px.line(plot_df, x="date", y=y, color=cat_field, title=colname, markers=True)

            fig.update_layout(xaxis=dict(type="date"))
            st.plotly_chart(fig, use_container_width=True)

        # === 表格 ===
        st.subheader("📋 Detailed Data")
        cols_to_show = ["date", cat_field]
        for sel in data_sel:
            if sel in mapping:
                cols_to_show.append(mapping[sel][0])

        st.dataframe(grouped[cols_to_show].assign(date=grouped["date"].dt.strftime("%Y-%m-%d")), use_container_width=True)

    else:
        st.info("👉 Please select **time range**, **data type**, and **categories** to generate the chart.")
