import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math


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


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    st.header("🧾 Sales Report by Category")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    # 🔹 确保 Datetime 是时间类型
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")

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

    # 应用时间范围筛选
    df_filtered = tx.copy()
    if start_date is not None and end_date is not None:
        mask = (df_filtered["Datetime"] >= pd.to_datetime(start_date)) & (
                df_filtered["Datetime"] <= pd.Timestamp(end_date))
        df_filtered = df_filtered.loc[mask]

    # ---------------- 使用与 high_level.py 一致的计算逻辑 ----------------
    df = df_filtered.copy()

    # 处理Tax列：移除$符号和逗号，转换为数字
    df["Tax"] = pd.to_numeric(
        df["Tax"].astype(str).str.replace(r'[^\d.-]', '', regex=True),
        errors="coerce"
    ).fillna(0)

    # 处理Gross Sales列
    df["Gross Sales"] = pd.to_numeric(df.get("Gross Sales"), errors="coerce").fillna(0.0)
    df["Qty"] = pd.to_numeric(df.get("Qty"), errors="coerce").fillna(0).abs()

    # 使用与 high_level.py 一致的计算逻辑：Daily Sales = Gross Sales - Tax
    df["Daily Sales"] = df["Gross Sales"] - df["Tax"]
    # 应用四舍五入到每行数据
    df["Daily Sales"] = df["Daily Sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    # ---------------- Bar Charts ----------------
    # 使用新的Daily Sales进行计算
    g = df.groupby("Category", as_index=False).agg(
        items_sold=("Qty", "sum"),
        daily_sales=("Daily Sales", "sum")
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
            # daily_sales 已经在行级别进行了四舍五入，这里只需要确保汇总正确
            st.plotly_chart(px.bar(g_chart, x="Category", y="daily_sales", title="Daily Sales (by Category)"),
                            use_container_width=True)
    else:
        st.info("No data under current filters.")
        return

    # ---------------- Group definitions ----------------
    bar_cats = ["Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"]
    retail_cats = [c for c in df["Category"].unique() if c not in bar_cats]

    # helper: 根据时间范围计算汇总数据 - 使用与 high_level.py 一致的逻辑
    def time_range_summary(data, cats, range_type, start_dt, end_dt):
        sub = data[data["Category"].isin(cats)].copy()
        if sub.empty:
            return pd.DataFrame()

        # 对于所有范围，直接汇总整个时间段的数据
        summary = sub.groupby("Category", as_index=False).agg(
            items_sold=("Qty", "sum"),
            daily_sales=("Daily Sales", "sum")
        )

        # 计算与前一个相同长度时间段的对比
        if start_dt and end_dt:
            time_diff = end_dt - start_dt
            prev_start = start_dt - time_diff
            prev_end = start_dt - timedelta(days=1)

            # 获取前一个时间段的数据
            prev_mask = (tx["Datetime"] >= prev_start) & (tx["Datetime"] <= prev_end)
            prev_data = tx.loc[prev_mask].copy()

            # 处理前一个时间段的数据 - 使用相同的计算逻辑
            if not prev_data.empty:
                prev_data["Tax"] = pd.to_numeric(
                    prev_data["Tax"].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors="coerce"
                ).fillna(0)
                prev_data["Gross Sales"] = pd.to_numeric(prev_data.get("Gross Sales"), errors="coerce").fillna(0.0)
                prev_data["Daily Sales"] = prev_data["Gross Sales"] - prev_data["Tax"]
                prev_data["Daily Sales"] = prev_data["Daily Sales"].apply(
                    lambda x: proper_round(x) if pd.notna(x) else x)

                prev_summary = prev_data[prev_data["Category"].isin(cats)].groupby("Category", as_index=False).agg(
                    prior_daily_sales=("Daily Sales", "sum")
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
    bar_df = time_range_summary(df, bar_cats, range_opt, start_date, end_date)
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

    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table + Multiselect ----------------
    st.subheader("📊 Retail Categories")

    # 🔹 使用与 high_level.py 一致的布局和格式
    col_retail, _ = st.columns([1, 2])
    with col_retail:
        all_retail_cats = sorted(df[df["Category"].isin(retail_cats)]["Category"].dropna().unique().tolist())
        sel_retail_cats = persisting_multiselect("Select Retail Categories", all_retail_cats, key="sr_retail_cats")

    retail_df = time_range_summary(df, retail_cats, range_opt, start_date, end_date)
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
    if not df[df["Category"].isin(retail_cats)].empty:
        retail_cats_summary = (df[df["Category"].isin(retail_cats)]
                               .groupby("Category")["Daily Sales"]
                               .sum()
                               .reset_index()
                               .sort_values("Daily Sales", ascending=False)
                               .head(9))

        # 对销售额进行四舍五入
        retail_cats_summary["Daily Sales"] = retail_cats_summary["Daily Sales"].apply(
            lambda x: proper_round(x) if pd.notna(x) else x
        )

        lines = []
        for i in range(0, len(retail_cats_summary), 3):
            chunk = retail_cats_summary.iloc[i:i + 3]
            line = " ".join([f"${int(v)} {n}" for n, v in zip(chunk["Category"], chunk["Daily Sales"])])
            lines.append(line)
        st.text("\n".join(lines))

    else:
        st.info("No retail categories available for comments.")