import streamlit as st
import pandas as pd
import plotly.express as px
from services.analytics import daily_summary_mongo, category_summary_mongo
from services.db import get_db

def _safe_sum(df, col):
    return df[col].sum() if col in df.columns else 0

def persisting_multiselect(label, options, key, default=None):
    """多选框 + 记忆功能"""
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options, default=st.session_state[key], key=key)

# ✅ 缓存数据库预聚合
@st.cache_data
def get_high_level_data(days=365):
    db = get_db()

    daily_docs = list(db.summary_daily.find({}, {"_id": 0}).limit(1))
    category_docs = list(db.summary_category.find({}, {"_id": 0}).limit(1))

    # ✅ 如果两个集合都是空的，立刻返回空 DataFrame
    if not daily_docs and not category_docs:
        return pd.DataFrame(), pd.DataFrame()

    daily = pd.DataFrame(list(db.summary_daily.find({}, {"_id": 0})))
    category = pd.DataFrame(list(db.summary_category.find({}, {"_id": 0})))

    if not daily.empty and "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

    if not category.empty and "date" in category.columns:
        category["date"] = pd.to_datetime(category["date"])
        category = category.sort_values(["Category", "date"])

    return daily, category


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    # === 加载 Mongo 聚合结果 ===
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

    all_cats = sorted(category["Category"].fillna("Unknown").unique().tolist())
    cats_sel = persisting_multiselect("Choose categories", all_cats, key="hl_cats")

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

        # ✅ 类别过滤
        grouped = grouped[grouped["Category"].isin(cats_sel)]

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
