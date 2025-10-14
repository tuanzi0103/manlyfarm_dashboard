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

    # 加载交易数据（包含日期信息）
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

    # 加载原始交易数据用于获取商品项（包含日期信息）
    item_sql = """
    SELECT 
        date(Datetime) as date,
        Category,
        Item,
        [Net Sales],
        Tax,
        Qty,
        [Gross Sales]
    FROM transactions
    WHERE Category IS NOT NULL AND Item IS NOT NULL
    """

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)
    items_df = pd.read_sql(item_sql, db)

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

    if not items_df.empty:
        items_df["date"] = pd.to_datetime(items_df["date"])
        # 移除缺失数据的日期 - 商品数据也过滤
        items_df = items_df[~items_df["date"].isin(pd.to_datetime(missing_dates))]

    return daily, category, items_df


def extract_item_name(item):
    """提取商品名称，移除毫升/升等容量信息"""
    if pd.isna(item):
        return item

    # 移除容量信息（数字后跟ml/L等）
    import re
    # 匹配数字后跟ml/L/升/毫升等模式
    pattern = r'\s*\d+\.?\d*\s*(ml|mL|L|升|毫升)\s*$'
    cleaned = re.sub(pattern, '', str(item), flags=re.IGNORECASE)

    # 移除首尾空格
    return cleaned.strip()


def prepare_sales_data(df_filtered):
    """使用与 high_level.py 相同的逻辑准备销售数据"""
    # 定义bar分类（与high_level.py一致）
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    # 复制数据避免修改原数据
    df = df_filtered.copy()

    # 对于bar分类，使用 net_sales_with_tax（已经包含税）
    # 对于非bar分类，使用 net_sales（不含税）
    df["final_sales"] = df.apply(
        lambda row: row["net_sales_with_tax"] if row["Category"] in bar_cats else row["net_sales"],
        axis=1
    )

    # 应用四舍五入（与high_level.py一致）
    df["final_sales"] = df["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
    df["qty"] = df["qty"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    return df


def calculate_item_sales(items_df, selected_categories, selected_items, start_date=None, end_date=None):
    """计算指定category和items的销售数据"""
    if not selected_categories or not selected_items:
        return pd.DataFrame()

    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 过滤指定category的商品
    filtered_items = filtered_items[filtered_items["Category"].isin(selected_categories)]

    # 清理商品名称用于匹配
    filtered_items["clean_item"] = filtered_items["Item"].apply(extract_item_name)

    # 应用商品项筛选
    filtered_items = filtered_items[filtered_items["clean_item"].isin(selected_items)]

    if filtered_items.empty:
        return pd.DataFrame()

    # 定义bar分类
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        if row["Category"] in bar_cats:
            # Bar分类：使用Net Sales + Tax
            tax_value = 0
            if pd.notna(row["Tax"]):
                try:
                    tax_str = str(row["Tax"]).replace('$', '').replace(',', '')
                    tax_value = float(tax_str) if tax_str else 0
                except:
                    tax_value = 0
            return proper_round(row["Net Sales"] + tax_value)
        else:
            # 非Bar分类：直接使用Net Sales
            return proper_round(row["Net Sales"])

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 按商品项汇总
    item_summary = filtered_items.groupby(["Category", "clean_item"]).agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    return item_summary.rename(columns={
        "clean_item": "Item",
        "Qty": "Sum of Items Sold",
        "final_sales": "Sum of Daily Sales"
    })[["Category", "Item", "Sum of Items Sold", "Sum of Daily Sales"]]


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    st.header("🧾 Sales Report by Category")

    # 预加载所有数据 - 使用与high_level.py相同的数据源
    with st.spinner("Loading data..."):
        daily, category_tx, items_df = preload_all_data()

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
            # 改为欧洲日期格式显示
            t1 = st.date_input(
                "From",
                value=pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
                key="sr_date_from",
                format="DD/MM/YYYY"
            )
        with col_to:
            # 改为欧洲日期格式显示
            t2 = st.date_input(
                "To",
                value=pd.Timestamp.today().normalize(),
                key="sr_date_to",
                format="DD/MM/YYYY"
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

    # 应用数据修复
    df_filtered_fixed = prepare_sales_data(df_filtered)

    # ---------------- Bar Charts ----------------
    # 使用修复后的数据
    g = df_filtered_fixed.groupby("Category", as_index=False).agg(
        items_sold=("qty", "sum"),
        daily_sales=("final_sales", "sum")  # 使用修复后的销售额
    ).sort_values("items_sold", ascending=False)

    if not g.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(g, x="Category", y="items_sold", title="Items Sold (by Category)"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(g, x="Category", y="daily_sales", title="Daily Sales (by Category)"),
                            use_container_width=True)
    else:
        st.info("No data under current filters.")
        return

    # ---------------- Group definitions ----------------
    # 使用与 high_level.py 完全相同的分类定义
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}
    retail_cats = [c for c in df_filtered_fixed["Category"].unique() if c not in bar_cats]

    # helper: 根据时间范围计算汇总数据 - 使用修复后的数据
    def time_range_summary(data, cats, range_type, start_dt, end_dt):
        sub = data[data["Category"].isin(cats)].copy()
        if sub.empty:
            return pd.DataFrame()

        # 使用修复后的数据聚合
        summary = sub.groupby("Category", as_index=False).agg(
            items_sold=("qty", "sum"),
            daily_sales=("final_sales", "sum")  # 使用修复后的销售额
        )

        # 计算与前一个相同长度时间段的对比
        if start_dt and end_dt:
            time_diff = end_dt - start_dt
            prev_start = start_dt - time_diff
            prev_end = start_dt - timedelta(days=1)

            # 获取前一个时间段的数据 - 使用相同的修复逻辑
            prev_mask = (category_tx["date"] >= prev_start) & (category_tx["date"] <= prev_end)
            prev_data = category_tx.loc[prev_mask].copy()

            # 对历史数据也应用相同的修复逻辑
            prev_data_fixed = prepare_sales_data(prev_data)

            if not prev_data_fixed.empty:
                prev_summary = prev_data_fixed[prev_data_fixed["Category"].isin(cats)].groupby("Category",
                                                                                               as_index=False).agg(
                    prior_daily_sales=("final_sales", "sum")  # 使用修复后的销售额
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

        # 计算日均销量 - 四舍五入保留整数
        if start_dt and end_dt:
            days_count = (end_dt - start_dt).days + 1
            summary["per_day"] = summary["items_sold"] / days_count
        else:
            summary["per_day"] = summary["items_sold"] / 7  # 默认按7天计算

        # 对 per_day 进行四舍五入保留整数
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
    bar_df = time_range_summary(df_filtered_fixed, bar_cats, range_opt, start_date, end_date)

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

        # Bar分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.subheader("📦 Bar Category Items")

        # 选择Bar分类 - 使用三列布局控制长度
        col_bar1, col_bar2, col_bar3 = st.columns([1, 1, 1])
        with col_bar1:
            bar_category_options = sorted(bar_df["Row Labels"].unique())
            selected_bar_categories = persisting_multiselect(
                "Select Bar Categories",
                options=bar_category_options,
                key="bar_categories_select"
            )

        # 根据选择的分类显示商品项
        if selected_bar_categories:
            # 获取选中分类的所有商品项
            bar_items_df = items_df[items_df["Category"].isin(selected_bar_categories)].copy()
            if not bar_items_df.empty:
                bar_items_df["clean_item"] = bar_items_df["Item"].apply(extract_item_name)
                bar_item_options = sorted(bar_items_df["clean_item"].dropna().unique())

                # 选择商品项 - 使用三列布局控制长度
                col_bar_items1, col_bar_items2, col_bar_items3 = st.columns([1, 1, 1])
                with col_bar_items1:
                    selected_bar_items = persisting_multiselect(
                        "Select Items from Bar Categories",
                        options=bar_item_options,
                        key="bar_items_select"
                    )

                # 显示选中的商品项数据
                if selected_bar_items:
                    bar_item_summary = calculate_item_sales(
                        items_df, selected_bar_categories, selected_bar_items, start_date, end_date
                    )

                    if not bar_item_summary.empty:
                        st.dataframe(bar_item_summary, use_container_width=True)

                        # 显示小计
                        total_qty = bar_item_summary["Sum of Items Sold"].sum()
                        total_sales = bar_item_summary["Sum of Daily Sales"].sum()
                        st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales}")

                        # 调试信息：显示数据条数
                        filtered_debug = items_df[
                            (items_df["Category"].isin(selected_bar_categories)) &
                            (items_df["Item"].apply(extract_item_name).isin(selected_bar_items))
                            ]
                        if start_date is not None and end_date is not None:
                            mask = (filtered_debug["date"] >= pd.to_datetime(start_date)) & (
                                    filtered_debug["date"] <= pd.Timestamp(end_date))
                            filtered_debug = filtered_debug.loc[mask]

                        st.write(f"**Debug:** Found {len(filtered_debug)} transaction records for selected criteria")
            else:
                st.info("No items found for selected Bar categories.")
    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table + Multiselect ----------------
    st.subheader("📊 Retail Categories")

    retail_df = time_range_summary(df_filtered_fixed, retail_cats, range_opt, start_date, end_date)

    if not retail_df.empty:
        retail_df = retail_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales": "Sum of Daily Sales",
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })
        retail_df["Weekly change"] = retail_df["Weekly change"].apply(format_change)

        st.dataframe(
            retail_df[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales", "Weekly change", "Per day"]]
            .style.applymap(highlight_change, subset=["Weekly change"]),
            use_container_width=True
        )

        # Retail分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.subheader("📦 Retail Category Items")

        # 选择Retail分类 - 使用三列布局控制长度
        col_retail1, col_retail2, col_retail3 = st.columns([1, 1, 1])
        with col_retail1:
            retail_category_options = sorted(retail_df["Row Labels"].unique())
            selected_retail_categories = persisting_multiselect(
                "Select Retail Categories",
                options=retail_category_options,
                key="retail_categories_select"
            )

        # 根据选择的分类显示商品项
        if selected_retail_categories:
            # 获取选中分类的所有商品项
            retail_items_df = items_df[items_df["Category"].isin(selected_retail_categories)].copy()
            if not retail_items_df.empty:
                retail_items_df["clean_item"] = retail_items_df["Item"].apply(extract_item_name)
                retail_item_options = sorted(retail_items_df["clean_item"].dropna().unique())

                # 选择商品项 - 使用三列布局控制长度
                col_retail_items1, col_retail_items2, col_retail_items3 = st.columns([1, 1, 1])
                with col_retail_items1:
                    selected_retail_items = persisting_multiselect(
                        "Select Items from Retail Categories",
                        options=retail_item_options,
                        key="retail_items_select"
                    )

                # 显示选中的商品项数据
                if selected_retail_items:
                    retail_item_summary = calculate_item_sales(
                        items_df, selected_retail_categories, selected_retail_items, start_date, end_date
                    )

                    if not retail_item_summary.empty:
                        st.dataframe(retail_item_summary, use_container_width=True)

                        # 显示小计
                        total_qty = retail_item_summary["Sum of Items Sold"].sum()
                        total_sales = retail_item_summary["Sum of Daily Sales"].sum()
                        st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales}")
            else:
                st.info("No items found for selected Retail categories.")
    else:
        st.info("No data for Retail categories.")

    # ---------------- Comment (Retail Top Categories) ----------------
    st.markdown("### 💬 Comment")
    if not df_filtered_fixed[df_filtered_fixed["Category"].isin(retail_cats)].empty:
        retail_cats_summary = (df_filtered_fixed[df_filtered_fixed["Category"].isin(retail_cats)]
                               .groupby("Category")["final_sales"]  # 使用修复后的销售额
                               .sum()
                               .reset_index()
                               .sort_values("final_sales", ascending=False)
                               .head(9))

        lines = []
        for i in range(0, len(retail_cats_summary), 3):
            chunk = retail_cats_summary.iloc[i:i + 3]
            line = " ".join([f"${int(v)} {n}" for n, v in zip(chunk["Category"], chunk["final_sales"])])
            lines.append(line)
        st.text("\n".join(lines))

    else:
        st.info("No retail categories available for comments.")