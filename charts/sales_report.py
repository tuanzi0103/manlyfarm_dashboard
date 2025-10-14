import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from services.db import get_db


def proper_round(x):
    """标准的四舍五入方法，0.5总是向上舍入"""
    if pd.isna(x):
        return x
    return math.floor(x + 0.5)


def persisting_multiselect(label, options, key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options, default=st.session_state[key], key=key)


def _safe_sum(df, col):
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return float(pd.to_numeric(s, errors="coerce").sum(skipna=True))
    s = (
        s.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return float(pd.to_numeric(s, errors="coerce").sum(skipna=True) or 0.0)


@st.cache_data(ttl=600, show_spinner=False)
def preload_all_data():
    """预加载所有需要的数据 - 与high_level.py相同的函数"""
    db = get_db()

    # 加载交易数据
    daily_sql = """
    WITH transaction_totals AS (
        SELECT 
            date(Datetime) AS date,
            [Transaction ID] AS txn_id,
            SUM([Gross Sales]) AS total_gross_sales,
            SUM(COALESCE(CAST(REPLACE(REPLACE([Tax], '$', ''), ',', '') AS REAL), 0)) AS total_tax,
            SUM(Qty) AS total_qty
        FROM transactions
        GROUP BY date, [Transaction ID]
    )
    SELECT
        date,
        SUM(ROUND(total_gross_sales - total_tax, 2)) AS net_sales_with_tax,
        SUM(total_gross_sales) AS gross_sales,
        SUM(total_tax) AS total_tax,
        COUNT(DISTINCT txn_id) AS transactions,
        CASE 
            WHEN COUNT(DISTINCT txn_id) > 0 
            THEN SUM(ROUND(total_gross_sales - total_tax, 2)) * 1.0 / COUNT(DISTINCT txn_id)
            ELSE 0 
        END AS avg_txn,
        SUM(total_qty) AS qty
    FROM transaction_totals
    GROUP BY date
    ORDER BY date;
    """

    category_sql = """
    WITH category_transactions AS (
        SELECT 
            date(Datetime) AS date,
            Category,
            [Transaction ID] AS txn_id,
            SUM([Net Sales]) AS cat_net_sales,
            SUM(COALESCE(CAST(REPLACE(REPLACE([Tax], '$', ''), ',', '') AS REAL), 0)) AS cat_tax,
            SUM([Gross Sales]) AS cat_gross,
            SUM(Qty) AS cat_qty
        FROM transactions
        GROUP BY date, Category, [Transaction ID]
    ),
    category_daily AS (
        SELECT
            date,
            Category,
            txn_id,
            SUM(ROUND(cat_net_sales + cat_tax, 2)) AS cat_total_with_tax,
            SUM(cat_net_sales) AS cat_net_sales,
            SUM(cat_tax) AS cat_tax,
            SUM(cat_gross) AS cat_gross,
            SUM(cat_qty) AS cat_qty
        FROM category_transactions
        GROUP BY date, Category, txn_id
    )
    SELECT
        date,
        Category,
        SUM(cat_total_with_tax) AS net_sales_with_tax,
        SUM(cat_net_sales) AS net_sales,
        SUM(cat_tax) AS total_tax,
        COUNT(DISTINCT txn_id) AS transactions,
        CASE 
            WHEN COUNT(DISTINCT txn_id) > 0 
            THEN SUM(cat_total_with_tax) * 1.0 / COUNT(DISTINCT txn_id)
            ELSE 0 
        END AS avg_txn,
        SUM(cat_gross) AS gross,
        SUM(cat_qty) AS qty
    FROM category_daily
    GROUP BY date, Category
    ORDER BY date, Category;
    """

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        # 移除缺失数据的日期 (8.18, 8.19, 8.20) - 所有数据都过滤
        missing_dates = ['2025-08-18', '2025-08-19', '2025-08-20']
        daily = daily[~daily["date"].isin(pd.to_datetime(missing_dates))]

    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])
        category = category.sort_values(["Category", "date"])

        # 移除缺失数据的日期 - 所有分类都过滤
        category = category[~category["date"].isin(pd.to_datetime(missing_dates))]

    return daily, category


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    st.header("🧾 Sales Report by Category")

    # 预加载所有数据 - 使用与high_level.py相同的数据源
    with st.spinner("Loading data..."):
        daily, category_tx = preload_all_data()

    if category_tx.empty:
        st.info("No category data available.")
        return

    # ---------------- Time Range Filter ----------------
    st.subheader("📅 Time Range")

    # 🔹 使用三列布局缩短下拉框宽度，与 high_level.py 保持一致
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        range_opt = st.selectbox("Select range", ["Custom dates", "WTD", "MTD", "YTD"], key="sr_range")

    today = pd.Timestamp.today().normalize()
    start_date, end_date = None, today

    if range_opt == "Custom dates":
        # 使用三列布局，与 "Select range" 一致
        col_from, col_to, _ = st.columns([1, 1, 1])
        with col_from:
            t1 = st.date_input(
                "From",
                value=pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
                key="sr_date_from"
            )
        with col_to:
            t2 = st.date_input(
                "To",
                value=pd.Timestamp.today().normalize(),
                key="sr_date_to"
            )
        if t1 and t2:
            start_date, end_date = pd.to_datetime(t1), pd.to_datetime(t2)
    elif range_opt == "WTD":
        start_date = today - pd.Timedelta(days=today.weekday())
    elif range_opt == "MTD":
        start_date = today.replace(day=1)
    elif range_opt == "YTD":
        start_date = today.replace(month=1, day=1)

    # 应用时间范围筛选到category数据
    df_filtered = category_tx.copy()
    if start_date is not None and end_date is not None:
        mask = (df_filtered["date"] >= pd.to_datetime(start_date)) & (
                df_filtered["date"] <= pd.Timestamp(end_date))
        df_filtered = df_filtered.loc[mask]

    # ---------------- Bar Charts ----------------
    # 使用high_level.py处理好的数据
    g = df_filtered.groupby("Category", as_index=False).agg(
        items_sold=("qty", "sum"),
        daily_sales=("net_sales_with_tax", "sum")
    ).sort_values("items_sold", ascending=False)

    if not g.empty:
        c1, c2 = st.columns(2)
        with c1:
            # 对items_sold进行四舍五入
            g_chart = g.copy()
            g_chart["items_sold"] = g_chart["items_sold"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            st.plotly_chart(px.bar(g_chart, x="Category", y="items_sold", title="Items Sold (by Category)"),
                            use_container_width=True)
        with c2:
            # daily_sales 使用high_level.py已经计算好的net_sales_with_tax
            g_chart["daily_sales"] = g_chart["daily_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            st.plotly_chart(px.bar(g_chart, x="Category", y="daily_sales", title="Daily Sales (by Category)"),
                            use_container_width=True)
    else:
        st.info("No data under current filters.")
        return

    # ---------------- Group definitions ----------------
    bar_cats = ["Cafe Drinks", "Smoothie Bar", "Smoothies", "Soups", "Sweet Treats", "Wraps & Salads"]
    retail_cats = [c for c in df_filtered["Category"].unique() if c not in bar_cats]

    # helper: 根据时间范围计算汇总数据 - 使用high_level.py处理好的数据
    def time_range_summary(data, cats, range_type, start_dt, end_dt):
        sub = data[data["Category"].isin(cats)].copy()
        if sub.empty:
            return pd.DataFrame()

        # 使用high_level.py处理好的数据直接聚合
        summary = sub.groupby("Category", as_index=False).agg(
            items_sold=("qty", "sum"),
            daily_sales=("net_sales_with_tax", "sum")
        )

        # 计算与前一个相同长度时间段的对比
        if start_dt and end_dt:
            time_diff = end_dt - start_dt
            prev_start = start_dt - time_diff
            prev_end = start_dt - timedelta(days=1)

            # 获取前一个时间段的数据 - 使用相同的high_level.py数据源
            prev_mask = (category_tx["date"] >= prev_start) & (category_tx["date"] <= prev_end)
            prev_data = category_tx.loc[prev_mask].copy()

            if not prev_data.empty:
                prev_summary = prev_data[prev_data["Category"].isin(cats)].groupby("Category", as_index=False).agg(
                    prior_daily_sales=("net_sales_with_tax", "sum")
                )

                summary = summary.merge(prev_summary, on="Category", how="left")
                summary["prior_daily_sales"] = summary["prior_daily_sales"].fillna(0)
            else:
                summary["prior_daily_sales"] = 0
        else:
            summary["prior_daily_sales"] = 0

        # 计算周变化
        MIN_BASE = 50
        summary["weekly_change"] = np.where(
            summary["prior_daily_sales"] > MIN_BASE,
            (summary["daily_sales"] - summary["prior_daily_sales"]) / summary["prior_daily_sales"],
            np.nan
        )

        # 计算日均销量
        if start_dt and end_dt:
            days_count = (end_dt - start_dt).days + 1
            summary["per_day"] = summary["items_sold"] / days_count
        else:
            summary["per_day"] = summary["items_sold"] / 7  # 默认按7天计算

        # 应用四舍五入
        summary["items_sold"] = summary["items_sold"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
        summary["daily_sales"] = summary["daily_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
        summary["per_day"] = summary["per_day"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        return summary

    # helper: 格式化 + 高亮
    def format_change(x):
        if pd.isna(x):
            return "N/A"
        return f"{x * 100:+.2f}%"

    def highlight_change(val):
        if val == "N/A":
            color = "gray"
        elif val.startswith("+"):
            color = "green"
        elif val.startswith("-"):
            color = "red"
        else:
            color = "black"
        return f"color: {color}"

    # ---------------- Bar table ----------------
    st.subheader("📊 Bar Categories")
    bar_df = time_range_summary(df_filtered, bar_cats, range_opt, start_date, end_date)
    if not bar_df.empty:
        bar_df = bar_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales": "Sum of Daily Sales",
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })
        bar_df["Weekly change"] = bar_df["Weekly change"].apply(format_change)

        st.dataframe(
            bar_df[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales", "Weekly change", "Per day"]]
            .style.applymap(highlight_change, subset=["Weekly change"]),
            use_container_width=True
        )

        # 调试信息：显示Smoothie Bar的总和
        smoothie_bar_total = bar_df[bar_df["Row Labels"] == "Smoothie Bar"]["Sum of Daily Sales"].sum()
        st.caption(f"Debug: Smoothie Bar total = ${proper_round(smoothie_bar_total)}")

    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table + Multiselect ----------------
    st.subheader("📊 Retail Categories")

    # 🔹 使用与 high_level.py 一致的布局和格式
    col_retail, _ = st.columns([1, 2])
    with col_retail:
        all_retail_cats = sorted(
            df_filtered[df_filtered["Category"].isin(retail_cats)]["Category"].dropna().unique().tolist())
        sel_retail_cats = persisting_multiselect("Select Retail Categories", all_retail_cats, key="sr_retail_cats")

    retail_df = time_range_summary(df_filtered, retail_cats, range_opt, start_date, end_date)
    if not retail_df.empty:
        retail_df = retail_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales": "Sum of Daily Sales",
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })
        if sel_retail_cats:
            retail_df = retail_df[retail_df["Row Labels"].isin(sel_retail_cats)]
        retail_df["Weekly change"] = retail_df["Weekly change"].apply(format_change)

        st.dataframe(
            retail_df[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales", "Weekly change", "Per day"]]
            .style.applymap(highlight_change, subset=["Weekly change"]),
            use_container_width=True
        )

    else:
        st.info("No data for Retail categories.")

    # ---------------- Comment (Retail Top Categories) ----------------
    st.markdown("### 💬 Comment")
    if not df_filtered[df_filtered["Category"].isin(retail_cats)].empty:
        retail_cats_summary = (df_filtered[df_filtered["Category"].isin(retail_cats)]
                               .groupby("Category")["net_sales_with_tax"]
                               .sum()
                               .reset_index()
                               .sort_values("net_sales_with_tax", ascending=False)
                               .head(9))

        # 对销售额进行四舍五入
        retail_cats_summary["net_sales_with_tax"] = retail_cats_summary["net_sales_with_tax"].apply(
            lambda x: proper_round(x) if pd.notna(x) else x
        )

        lines = []
        for i in range(0, len(retail_cats_summary), 3):
            chunk = retail_cats_summary.iloc[i:i + 3]
            line = " ".join([f"${int(v)} {n}" for n, v in zip(chunk["Category"], chunk["net_sales_with_tax"])])
            lines.append(line)
        st.text("\n".join(lines))

    else:
        st.info("No retail categories available for comments.")