import streamlit as st
import pandas as pd
import plotly.express as px
import math
import numpy as np
from services.db import get_db


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


def proper_round(x):
    """标准的四舍五入方法，0.5总是向上舍入"""
    if pd.isna(x):
        return x
    return math.floor(x + 0.5)


def persisting_multiselect(label, options, key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options, default=st.session_state[key], key=key)


# === 修正的聚合逻辑 - 确保bar计算正确 ===
@st.cache_data
def get_high_level_data():
    db = get_db()

    # === total（日级）：SUM(ROUND(Gross - Tax, 2))，再汇总 ===
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
        -- ✅ 修正 total 逻辑：逐行 (Gross - Tax) 保留两位小数后求和
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

    # === category（日级）：SUM(ROUND(Net + Tax, 2)) ===
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
        -- ✅ 逐行保留两位小数后再汇总
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
    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])

    return daily, category


def _prepare_inventory_grouped(inv: pd.DataFrame):
    if inv is None or inv.empty:
        return pd.DataFrame(), None

    df = inv.copy()

    if "source_date" in df.columns:
        df["date"] = pd.to_datetime(df["source_date"], errors="coerce")
    else:
        return pd.DataFrame(), None

    # Category 列
    if "Categories" in df.columns:
        df["Category"] = df["Categories"].astype(str)
    elif "Category" in df.columns:
        df["Category"] = df["Category"].astype(str)
    else:
        df["Category"] = "Unknown"

    # === 用 catalogue 现算 - 应用新的inventory value计算逻辑 ===
    # 1. 过滤掉 Current Quantity Vie Market & Bar 为负数或0的行
    df["Quantity"] = pd.to_numeric(df["Current Quantity Vie Market & Bar"], errors="coerce")
    mask = (df["Quantity"] > 0)  # 只保留正数
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame(), None

    # 2. 把 Default Unit Cost 为空的值补为0
    df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce").fillna(0)

    # 3. 计算 inventory value: Default Unit Cost * Current Quantity Vie Market & Bar
    df["Inventory Value"] = df["UnitCost"] * df["Quantity"]

    # 四舍五入保留整数
    df["Inventory Value"] = df["Inventory Value"].apply(lambda x: proper_round(x) if not pd.isna(x) else 0)

    # 保留其他计算（如果需要）
    df["Price"] = pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0)

    # 修复：检查 TaxFlag 列是否存在，如果不存在则创建默认值
    if "TaxFlag" not in df.columns:
        df["TaxFlag"] = "N"  # 默认值，假设不含税

    def calc_retail(row):
        try:
            O, AA, tax = row["Price"], row["Quantity"], row["TaxFlag"]
            return (O / 11 * 10) * AA if tax == "Y" else O * AA
        except KeyError:
            # 如果列不存在，直接计算 Price * Quantity
            return row["Price"] * row["Quantity"]

    df["Retail Total"] = df.apply(calc_retail, axis=1)
    df["Profit"] = df["Retail Total"] - df["Inventory Value"]

    # 聚合
    g = (
        df.groupby(["date", "Category"], as_index=False)[["Inventory Value", "Profit"]]
        .sum(min_count=1)
    )

    latest_date = g["date"].max() if not g.empty else None
    return g, latest_date

# 移除 @st.cache_data 装饰器，因为函数中包含 widgets
def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    daily, category_tx = get_high_level_data()
    inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv)

    if daily.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    # === 特定日期选择 ===
    st.subheader("📅 Select Specific Date")
    col_date, _ = st.columns([1, 2])  # 第一个列放选择框，第二个留空拉窄宽度
    with col_date:
        available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
        selected_date = st.selectbox("Choose a specific date to view data", available_dates)

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    # 筛选选定日期的数据
    df_selected_date = daily[daily["date"] == selected_date_ts]

    today = pd.Timestamp.today().normalize()
    latest_date_tx = daily["date"].max()
    df_latest_tx = daily[daily["date"] == latest_date_tx]

    # === 计算客户数量 ===
    def calculate_customer_count(tx_df, selected_date):
        if tx_df is None or tx_df.empty:
            return 0
        if 'Datetime' not in tx_df.columns:
            return 0

        tx_df = tx_df.copy()
        tx_df['Datetime'] = pd.to_datetime(tx_df['Datetime'], errors='coerce')
        tx_df = tx_df.dropna(subset=['Datetime'])
        if tx_df.empty:
            return 0

        selected_date_str = selected_date.strftime('%Y-%m-%d')
        daily_tx = tx_df[tx_df['Datetime'].dt.strftime('%Y-%m-%d') == selected_date_str]
        if daily_tx.empty:
            return 0

        if 'Card Brand' not in daily_tx.columns or 'PAN Suffix' not in daily_tx.columns:
            return 0

        filtered_tx = daily_tx.dropna(subset=['Card Brand', 'PAN Suffix'])
        if filtered_tx.empty:
            return 0

        filtered_tx['Card Brand'] = filtered_tx['Card Brand'].str.title()
        filtered_tx['PAN Suffix'] = filtered_tx['PAN Suffix'].astype(str).str.split('.').str[0]
        unique_customers = filtered_tx[['Card Brand', 'PAN Suffix']].drop_duplicates()

        return len(unique_customers)

    # === KPI（交易，口径按小票） ===
    # 使用选定日期的数据 - 确保使用 net_sales_with_tax (Gross Sales - Tax)
    kpis_main = {
        "Daily Net Sales": proper_round(df_selected_date["net_sales_with_tax"].sum()),
        "Daily Transactions": df_selected_date["transactions"].sum(),
        "Number of Customers": calculate_customer_count(tx, selected_date),
        "Avg Transaction": df_selected_date["avg_txn"].mean(),
        "3M Avg": proper_round(daily["net_sales_with_tax"].rolling(90, min_periods=1).mean().iloc[-1]),
        "6M Avg": proper_round(daily["net_sales_with_tax"].rolling(180, min_periods=1).mean().iloc[-1]),
        "Items Sold": df_selected_date["qty"].sum(),
    }

    # === KPI（库存派生，catalogue-only） ===
    inv_value_latest = 0.0
    profit_latest = 0.0
    if inv_grouped is not None and not inv_grouped.empty and inv_latest_date is not None:
        sub = inv_grouped[inv_grouped["date"] == inv_latest_date]
        inv_value_latest = float(pd.to_numeric(sub["Inventory Value"], errors="coerce").sum())
        profit_latest = float(pd.to_numeric(sub["Profit"], errors="coerce").sum())

    st.markdown(f"### 📅 Selected Date: {selected_date}")
    labels_values = list(kpis_main.items()) + [
        ("Inventory Value", inv_value_latest),
        ("Profit (Amount)", profit_latest),
    ]
    captions = {
        "Inventory Value": f"as of {pd.to_datetime(inv_latest_date).strftime('%Y-%m-%d') if inv_latest_date else '-'}",
        "Profit (Amount)": f"as of {pd.to_datetime(inv_latest_date).strftime('%Y-%m-%d') if inv_latest_date else '-'}",
    }

    for row in range(0, len(labels_values), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row + i
            if idx < len(labels_values):
                label, val = labels_values[idx]
                # 使用标准的四舍五入方法
                if pd.isna(val):
                    display = "-"
                else:
                    # 去掉美元符号，并为 Avg Transaction 添加两位小数
                    if label == "Avg Transaction":
                        display = f"${val:,.2f}"
                    elif label in ["Daily Net Sales", "3M Avg", "6M Avg", "Inventory Value", "Profit (Amount)"]:
                        display = f"${proper_round(val):,}"
                    else:
                        display = f"{proper_round(val):,}"
                with col:
                    st.markdown(f"<div style='font-size:28px; font-weight:600'>{display}</div>", unsafe_allow_html=True)
                    st.caption(label)
                    if label in captions:
                        st.caption(captions[label])

    st.markdown("---")

    # === 交互选择 ===
    st.subheader("🔍 Select Parameters")

    # 🔹 用三列布局缩短下拉框宽度
    col1, col2, col3 = st.columns([1, 1, 1])

    # === 第一列：时间范围 ===
    with col1:
        time_range_options = ["Custom dates", "WTD", "MTD", "YTD"]
        time_range = st.multiselect("Choose time range", time_range_options, key="hl_time")

    # === 第二列：数据类型 ===
    with col2:
        data_options = [
            "Daily Net Sales", "Daily Transactions", "Avg Transaction", "3M Avg", "6M Avg",
            "Inventory Value", "Profit (Amount)", "Items Sold"
        ]
        data_sel = persisting_multiselect("Choose data type", data_options, key="hl_data")

    # === 第三列：分类 ===
    with col3:
        bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

        if category_tx is None or category_tx.empty:
            st.info("No category breakdown available.")
            return

        all_cats_tx = sorted(category_tx["Category"].fillna("Unknown").unique().tolist())

        # 调整选项顺序：bar, retail, total 在最上面，然后是其他类别
        special_cats = ["bar", "retail", "total"]
        all_cats_extended = special_cats + sorted([c for c in all_cats_tx if c not in special_cats])

        cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    # === 自定义日期范围选择 ===
    custom_dates_selected = False
    t1 = None
    t2 = None

    if "Custom dates" in time_range:
        custom_dates_selected = True
        # 🔹 修复1：使用与上面三列布局相同长度的列布局，与 sales_report.py 保持一致
        st.markdown("#### 📅 Custom Date Range")
        col_from, col_to, _ = st.columns([1, 1, 1])  # 改为三列布局
        with col_from:
            t1 = st.date_input(
                "From",
                value=pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
                key="date_from"
            )
        with col_to:
            t2 = st.date_input(
                "To",
                value=pd.Timestamp.today().normalize(),
                key="date_to"
            )

    # 修复1：修正条件判断逻辑
    has_time_range = bool(time_range)
    has_data_sel = bool(data_sel)
    has_cats_sel = bool(cats_sel)

    # 对于 Custom dates，需要确保日期已选择
    if "Custom dates" in time_range:
        has_valid_custom_dates = (t1 is not None and t2 is not None)
    else:
        has_valid_custom_dates = True

    if has_time_range and has_data_sel and has_cats_sel and has_valid_custom_dates:
        # 首先获取完整的数据用于计算滚动平均
        daily_full = daily.copy()
        grouped_tx_full = category_tx.copy()

        # 获取当前日期
        today = pd.Timestamp.today().normalize()

        # 计算时间范围筛选条件
        start_of_week = today - pd.Timedelta(days=today.weekday())
        start_of_month = today.replace(day=1)
        start_of_year = today.replace(month=1, day=1)

        # 应用时间范围筛选到daily数据
        daily_filtered = daily.copy()
        grouped_tx = category_tx.copy()

        if "WTD" in time_range:
            daily_filtered = daily_filtered[daily_filtered["date"] >= start_of_week]
            grouped_tx = grouped_tx[grouped_tx["date"] >= start_of_week]
        if "MTD" in time_range:
            daily_filtered = daily_filtered[daily_filtered["date"] >= start_of_month]
            grouped_tx = grouped_tx[grouped_tx["date"] >= start_of_month]
        if "YTD" in time_range:
            daily_filtered = daily_filtered[daily_filtered["date"] >= start_of_year]
            grouped_tx = grouped_tx[grouped_tx["date"] >= start_of_year]
        if custom_dates_selected and t1 and t2:
            t1_ts = pd.to_datetime(t1)
            t2_ts = pd.to_datetime(t2)
            daily_filtered = daily_filtered[
                (daily_filtered["date"] >= t1_ts) & (daily_filtered["date"] <= t2_ts)]
            grouped_tx = grouped_tx[
                (grouped_tx["date"] >= t1_ts) & (grouped_tx["date"] <= t2_ts)]

        grouped_inv = inv_grouped.copy()
        # 对库存数据应用相同的时间范围筛选
        if not grouped_inv.empty:
            if "WTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= start_of_week]
            if "MTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= start_of_month]
            if "YTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= start_of_year]
            if custom_dates_selected and t1 and t2:
                grouped_inv = grouped_inv[
                    (grouped_inv["date"] >= pd.to_datetime(t1)) & (grouped_inv["date"] <= pd.to_datetime(t2))]

        small_cats = [c for c in cats_sel if c not in ("bar", "retail", "total")]
        parts_tx = []

        if small_cats:
            parts_tx.append(grouped_tx[grouped_tx["Category"].isin(small_cats)])

        # === 应用新的计算逻辑 ===
        if "bar" in cats_sel:
            bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}
            bar_tx = grouped_tx[grouped_tx["Category"].isin(bar_cats)].copy()
            if not bar_tx.empty:
                # 修改：先将五类数据按日期聚合，再设置为bar
                bar_tx_aggregated = bar_tx.groupby("date").agg({
                    "net_sales_with_tax": "sum",
                    "net_sales": "sum",
                    "total_tax": "sum",
                    "transactions": "sum",
                    "avg_txn": "mean",
                    "gross": "sum",
                    "qty": "sum"
                }).reset_index()
                bar_tx_aggregated["Category"] = "bar"
                parts_tx.append(bar_tx_aggregated)

        if "retail" in cats_sel:
            retail_cats = {"Retail"}
            retail_tx = grouped_tx[grouped_tx["Category"].isin(retail_cats)].copy()
            if not retail_tx.empty:
                retail_tx["Category"] = "retail"
                parts_tx.append(retail_tx)

        if "total" in cats_sel:
            total_tx = daily_filtered.copy()
            total_tx["Category"] = "total"
            parts_tx.append(total_tx)

        if not parts_tx:
            st.warning("No data for selected categories.")
            return

        df_plot = pd.concat(parts_tx, ignore_index=True)

        # === 数据映射 ===
        data_map = {
            "Daily Net Sales": "net_sales_with_tax",
            "Daily Transactions": "transactions",
            "Avg Transaction": "avg_txn",
            "3M Avg": "net_sales_with_tax",
            "6M Avg": "net_sales_with_tax",
            "Items Sold": "qty",
        }

        # 处理滚动平均
        if "3M Avg" in data_sel or "6M Avg" in data_sel:
            # 为每个类别计算滚动平均
            df_plot_rolling = df_plot.copy()
            df_plot_rolling = df_plot_rolling.sort_values(["Category", "date"])

            # 使用完整数据计算滚动平均
            df_full_rolling = pd.concat([
                # 修改：bar类别需要先聚合五类数据
                grouped_tx_full[grouped_tx_full["Category"].isin(bar_cats)].groupby("date").agg({
                    "net_sales_with_tax": "sum",
                    "net_sales": "sum",
                    "total_tax": "sum",
                    "transactions": "sum",
                    "avg_txn": "mean",
                    "gross": "sum",
                    "qty": "sum"
                }).reset_index().assign(Category="bar") if cat == "bar" else
                grouped_tx_full.assign(Category="retail") if cat == "retail" else
                daily_full.assign(Category="total") if cat == "total" else
                grouped_tx_full[grouped_tx_full["Category"] == cat].copy()
                for cat in cats_sel
            ], ignore_index=True)

            df_full_rolling = df_full_rolling.sort_values(["Category", "date"])

            # 计算滚动平均
            window_3m = 90
            window_6m = 180

            df_full_rolling["3M Avg"] = df_full_rolling.groupby("Category")["net_sales_with_tax"].transform(
                lambda x: x.rolling(window_3m, min_periods=1).mean()
            )
            df_full_rolling["6M Avg"] = df_full_rolling.groupby("Category")["net_sales_with_tax"].transform(
                lambda x: x.rolling(window_6m, min_periods=1).mean()
            )

            # 应用时间范围筛选到滚动平均数据
            df_full_rolling_filtered = df_full_rolling.copy()
            if "WTD" in time_range:
                df_full_rolling_filtered = df_full_rolling_filtered[df_full_rolling_filtered["date"] >= start_of_week]
            if "MTD" in time_range:
                df_full_rolling_filtered = df_full_rolling_filtered[df_full_rolling_filtered["date"] >= start_of_month]
            if "YTD" in time_range:
                df_full_rolling_filtered = df_full_rolling_filtered[df_full_rolling_filtered["date"] >= start_of_year]
            if custom_dates_selected and t1 and t2:
                df_full_rolling_filtered = df_full_rolling_filtered[
                    (df_full_rolling_filtered["date"] >= pd.to_datetime(t1)) &
                    (df_full_rolling_filtered["date"] <= pd.to_datetime(t2))
                    ]

            # 合并滚动平均数据到主数据框
            df_plot = df_plot.merge(
                df_full_rolling_filtered[["date", "Category", "3M Avg", "6M Avg"]],
                on=["date", "Category"], how="left"
            )

        # 处理库存数据
        if "Inventory Value" in data_sel or "Profit (Amount)" in data_sel:
            if grouped_inv is not None and not grouped_inv.empty:
                # 确保库存数据有相同的列结构
                grouped_inv_plot = grouped_inv.copy()
                # 重命名列以匹配交易数据
                grouped_inv_plot = grouped_inv_plot.rename(columns={
                    "Inventory Value": "inventory_value",
                    "Profit": "profit_amount"
                })
                # 添加缺失的列
                for col in ["net_sales_with_tax", "transactions", "avg_txn", "qty"]:
                    grouped_inv_plot[col] = 0

                # 如果选择了库存相关的数据，将库存数据合并到主数据框
                if small_cats:
                    inv_small = grouped_inv_plot[grouped_inv_plot["Category"].isin(small_cats)]
                    df_plot = pd.concat([df_plot, inv_small], ignore_index=True)

                if "bar" in cats_sel:
                    bar_inv = grouped_inv_plot[grouped_inv_plot["Category"].isin(bar_cats)].copy()
                    if not bar_inv.empty:
                        bar_inv["Category"] = "bar"
                        df_plot = pd.concat([df_plot, bar_inv], ignore_index=True)

                if "retail" in cats_sel:
                    retail_inv = grouped_inv_plot[grouped_inv_plot["Category"].isin(["Retail"])].copy()
                    if not retail_inv.empty:
                        retail_inv["Category"] = "retail"
                        df_plot = pd.concat([df_plot, retail_inv], ignore_index=True)

                if "total" in cats_sel:
                    total_inv = grouped_inv_plot.copy()
                    total_inv_sum = total_inv.groupby("date").agg({
                        "inventory_value": "sum",
                        "profit_amount": "sum"
                    }).reset_index()
                    total_inv_sum["Category"] = "total"
                    # 添加缺失的列
                    for col in ["net_sales_with_tax", "transactions", "avg_txn", "qty"]:
                        total_inv_sum[col] = 0
                    df_plot = pd.concat([df_plot, total_inv_sum], ignore_index=True)

        # 确保数据列存在
        for col in data_map.values():
            if col not in df_plot.columns:
                df_plot[col] = 0

        # 添加库存数据列
        if "inventory_value" not in df_plot.columns:
            df_plot["inventory_value"] = 0
        if "profit_amount" not in df_plot.columns:
            df_plot["profit_amount"] = 0

        # 扩展数据映射
        data_map_extended = {
            **data_map,
            "Inventory Value": "inventory_value",
            "Profit (Amount)": "profit_amount"
        }

        # 🔹 修复2：把所有data type都展示在同一个折线图里
        if data_sel:
            # 创建融合数据框，将所有选中的数据列合并到一个图中
            melted_dfs = []

            for data_type in data_sel:
                if data_type not in data_map_extended:
                    continue

                col_name = data_map_extended[data_type]
                if col_name not in df_plot.columns:
                    continue

                # 为每个数据类型创建子数据框
                temp_df = df_plot[["date", "Category", col_name]].copy()
                temp_df = temp_df.rename(columns={col_name: "value"})
                temp_df["data_type"] = data_type

                # 过滤掉没有数据的行
                temp_df = temp_df[temp_df["value"].notna() & (temp_df["value"] != 0)]

                if not temp_df.empty:
                    melted_dfs.append(temp_df)

            if melted_dfs:
                # 合并所有数据
                combined_df = pd.concat(melted_dfs, ignore_index=True)

                # 创建图表 - 使用 data_type 和 Category 的组合作为线条
                combined_df["series"] = combined_df["Category"] + " - " + combined_df["data_type"]

                fig = px.line(
                    combined_df,
                    x="date",
                    y="value",
                    color="series",
                    title="All Selected Data Types by Category",
                    labels={"date": "Date", "value": "Value", "series": "Series"}
                )

                fig.update_layout(
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    hovermode="x unified",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # 显示数据表格
                with st.expander("View combined data for all selected types"):
                    display_df = combined_df.copy()
                    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                    display_df = display_df.rename(columns={
                        "date": "Date",
                        "Category": "Category",
                        "data_type": "Data Type",
                        "value": "Value"
                    })
                    display_df = display_df.sort_values(["Date", "Category", "Data Type"])
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No data available for the selected data types.")