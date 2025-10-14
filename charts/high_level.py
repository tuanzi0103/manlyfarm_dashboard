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


# === 预加载所有数据 ===
@st.cache_data(ttl=600, show_spinner=False)
def preload_all_data():
    """预加载所有需要的数据"""
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

        # 计算滚动平均值 - 使用更准确的窗口计算
        daily["3M_Avg_Rolling"] = daily["net_sales_with_tax"].rolling(window=90, min_periods=1, center=False).mean()
        daily["6M_Avg_Rolling"] = daily["net_sales_with_tax"].rolling(window=180, min_periods=1, center=False).mean()

    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])
        category = category.sort_values(["Category", "date"])

        # 移除缺失数据的日期 - 所有分类都过滤
        category = category[~category["date"].isin(pd.to_datetime(missing_dates))]

    return daily, category


@st.cache_data(ttl=600, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def prepare_chart_data_fast(daily, category_tx, inv_grouped, time_range, data_sel, cats_sel,
                            custom_dates_selected=False, t1=None, t2=None):
    """快速准备图表数据"""
    if not time_range or not data_sel or not cats_sel:
        return None

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

    # 定义bar分类
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    # 处理bar分类 - 重新计算bar的滚动平均
    if "bar" in cats_sel:
        bar_tx = grouped_tx[grouped_tx["Category"].isin(bar_cats)].copy()
        if not bar_tx.empty:
            # 先按日期聚合bar数据
            bar_daily_agg = bar_tx.groupby("date").agg({
                "net_sales_with_tax": "sum",
                "transactions": "sum",
                "qty": "sum"
            }).reset_index()

            # 计算bar的平均交易额
            bar_daily_agg["avg_txn"] = bar_daily_agg.apply(
                lambda x: x["net_sales_with_tax"] / x["transactions"] if x["transactions"] > 0 else 0,
                axis=1
            )

            # 为bar数据计算准确的滚动平均
            bar_daily_agg["3M_Avg_Rolling"] = bar_daily_agg["net_sales_with_tax"].rolling(window=90, min_periods=1,
                                                                                          center=False).mean()
            bar_daily_agg["6M_Avg_Rolling"] = bar_daily_agg["net_sales_with_tax"].rolling(window=180, min_periods=1,
                                                                                          center=False).mean()

            bar_daily_agg["Category"] = "bar"
            parts_tx.append(bar_daily_agg)

    # 处理retail分类 = total - bar
    if "retail" in cats_sel:
        # 获取每日total数据
        total_daily = daily_filtered.copy()
        total_daily = total_daily.rename(columns={
            "net_sales_with_tax": "total_net_sales",
            "transactions": "total_transactions",
            "avg_txn": "total_avg_txn",
            "qty": "total_qty"
        })

        # 获取每日bar数据
        bar_daily = grouped_tx[grouped_tx["Category"].isin(bar_cats)].groupby("date").agg({
            "net_sales_with_tax": "sum",
            "transactions": "sum",
            "qty": "sum"
        }).reset_index()
        bar_daily = bar_daily.rename(columns={
            "net_sales_with_tax": "bar_net_sales",
            "transactions": "bar_transactions",
            "qty": "bar_qty"
        })

        # 合并total和bar数据
        retail_data = total_daily.merge(bar_daily, on="date", how="left")

        # 计算retail = total - bar
        retail_data["net_sales_with_tax"] = retail_data["total_net_sales"] - retail_data["bar_net_sales"].fillna(0)
        retail_data["transactions"] = retail_data["total_transactions"] - retail_data["bar_transactions"].fillna(0)
        retail_data["qty"] = retail_data["total_qty"] - retail_data["bar_qty"].fillna(0)

        # 计算平均交易额
        retail_data["avg_txn"] = retail_data.apply(
            lambda x: x["net_sales_with_tax"] / x["transactions"] if x["transactions"] > 0 else 0,
            axis=1
        )

        # 为retail数据计算准确的滚动平均
        retail_data["3M_Avg_Rolling"] = retail_data["net_sales_with_tax"].rolling(window=90, min_periods=1,
                                                                                  center=False).mean()
        retail_data["6M_Avg_Rolling"] = retail_data["net_sales_with_tax"].rolling(window=180, min_periods=1,
                                                                                  center=False).mean()

        # 只保留需要的列
        retail_tx = retail_data[
            ["date", "net_sales_with_tax", "transactions", "avg_txn", "qty", "3M_Avg_Rolling", "6M_Avg_Rolling"]].copy()
        retail_tx["Category"] = "retail"
        parts_tx.append(retail_tx)

    if "total" in cats_sel:
        total_tx = daily_filtered.copy()
        total_tx["Category"] = "total"
        parts_tx.append(total_tx)

    if not parts_tx:
        return None

    df_plot = pd.concat(parts_tx, ignore_index=True)

    # 数据映射
    data_map_extended = {
        "Daily Net Sales": "net_sales_with_tax",
        "Daily Transactions": "transactions",
        "Avg Transaction": "avg_txn",
        "3M Avg": "3M_Avg_Rolling",
        "6M Avg": "6M_Avg_Rolling",
        "Items Sold": "qty",
        "Inventory Value": "inventory_value",
        "Profit (Amount)": "profit_amount"
    }

    # 处理库存数据
    if any(data in ["Inventory Value", "Profit (Amount)"] for data in data_sel):
        if not grouped_inv.empty:
            grouped_inv_plot = grouped_inv.copy()
            grouped_inv_plot = grouped_inv_plot.rename(columns={
                "Inventory Value": "inventory_value",
                "Profit": "profit_amount"
            })
            # 添加缺失的列
            for col in ["net_sales_with_tax", "transactions", "avg_txn", "qty", "3M_Avg_Rolling", "6M_Avg_Rolling"]:
                grouped_inv_plot[col] = 0

            # 合并库存数据
            if small_cats:
                inv_small = grouped_inv_plot[grouped_inv_plot["Category"].isin(small_cats)]
                df_plot = pd.concat([df_plot, inv_small], ignore_index=True)

            if "bar" in cats_sel:
                bar_inv = grouped_inv_plot[grouped_inv_plot["Category"].isin(bar_cats)].copy()
                if not bar_inv.empty:
                    bar_inv["Category"] = "bar"
                    df_plot = pd.concat([df_plot, bar_inv], ignore_index=True)

            if "retail" in cats_sel:
                retail_inv = grouped_inv_plot[grouped_inv_plot["Category"] == "Retail"].copy()
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
                for col in ["net_sales_with_tax", "transactions", "avg_txn", "qty", "3M_Avg_Rolling", "6M_Avg_Rolling"]:
                    total_inv_sum[col] = 0
                df_plot = pd.concat([df_plot, total_inv_sum], ignore_index=True)

    # 确保所有需要的列都存在
    for col_name in data_map_extended.values():
        if col_name not in df_plot.columns:
            df_plot[col_name] = 0

    # 添加库存数据列
    if "inventory_value" not in df_plot.columns:
        df_plot["inventory_value"] = 0
    if "profit_amount" not in df_plot.columns:
        df_plot["profit_amount"] = 0

    # 创建融合数据框用于图表
    melted_dfs = []
    for data_type in data_sel:
        col_name = data_map_extended.get(data_type)
        if col_name and col_name in df_plot.columns:
            temp_df = df_plot[["date", "Category", col_name]].copy()
            temp_df = temp_df.rename(columns={col_name: "value"})
            temp_df["data_type"] = data_type

            # 对 Daily Net Sales 进行四舍五入取整
            if data_type == "Daily Net Sales":
                temp_df["value"] = temp_df["value"].apply(lambda x: proper_round(x) if not pd.isna(x) else 0)

            # 放宽过滤条件
            temp_df = temp_df[temp_df["value"].notna()]
            if not temp_df.empty:
                melted_dfs.append(temp_df)

    if melted_dfs:
        combined_df = pd.concat(melted_dfs, ignore_index=True)
        combined_df["series"] = combined_df["Category"] + " - " + combined_df["data_type"]

        # 确保最终数据中完全移除缺失日期的数据点
        missing_dates = ['2025-08-18', '2025-08-19', '2025-08-20']
        combined_df = combined_df[~combined_df["date"].isin(pd.to_datetime(missing_dates))]

        return combined_df

    return None


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    # 预加载所有数据
    with st.spinner("Loading data..."):
        daily, category_tx = preload_all_data()
        inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv)

    if daily.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    # === 特定日期选择 ===
    st.subheader("📅 Select Specific Date")
    col_date, _ = st.columns([1, 2])
    with col_date:
        available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
        # 将日期格式改为欧洲格式显示
        available_dates_formatted = [date.strftime('%d/%m/%Y') for date in available_dates]
        selected_date_formatted = st.selectbox("Choose a specific date to view data", available_dates_formatted)

        # 将选择的日期转换回日期对象
        selected_date = pd.to_datetime(selected_date_formatted, format='%d/%m/%Y').date()

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    # 筛选选定日期的数据
    df_selected_date = daily[daily["date"].dt.date == selected_date]

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
    kpis_main = {
        "Daily Net Sales": proper_round(df_selected_date["net_sales_with_tax"].sum()),
        "Daily Transactions": df_selected_date["transactions"].sum(),
        "Number of Customers": calculate_customer_count(tx, selected_date),
        "Avg Transaction": df_selected_date["avg_txn"].mean(),
        "3M Avg": proper_round(daily["3M_Avg_Rolling"].iloc[-1]),
        "6M Avg": proper_round(daily["6M_Avg_Rolling"].iloc[-1]),
        "Items Sold": df_selected_date["qty"].sum(),
    }

    # === KPI（库存派生，catalogue-only） ===
    inv_value_latest = 0.0
    profit_latest = 0.0
    if inv_grouped is not None and not inv_grouped.empty and inv_latest_date is not None:
        sub = inv_grouped[inv_grouped["date"] == inv_latest_date]
        inv_value_latest = float(pd.to_numeric(sub["Inventory Value"], errors="coerce").sum())
        profit_latest = float(pd.to_numeric(sub["Profit"], errors="coerce").sum())

    st.markdown(f"### 📅 Selected Date: {selected_date.strftime('%d/%m/%Y')}")  # 改为欧洲日期格式
    labels_values = list(kpis_main.items()) + [
        ("Inventory Value", inv_value_latest),
        ("Profit (Amount)", profit_latest),
    ]
    captions = {
        "Inventory Value": f"as of {pd.to_datetime(inv_latest_date).strftime('%d/%m/%Y') if inv_latest_date else '-'}",
        # 改为欧洲日期格式
        "Profit (Amount)": f"as of {pd.to_datetime(inv_latest_date).strftime('%d/%m/%Y') if inv_latest_date else '-'}",
        # 改为欧洲日期格式
    }

    for row in range(0, len(labels_values), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row + i
            if idx < len(labels_values):
                label, val = labels_values[idx]
                if pd.isna(val):
                    display = "-"
                else:
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
        if category_tx is None or category_tx.empty:
            st.info("No category breakdown available.")
            return

        all_cats_tx = sorted(category_tx["Category"].fillna("Unknown").unique().tolist())
        special_cats = ["bar", "retail", "total"]
        all_cats_extended = special_cats + sorted([c for c in all_cats_tx if c not in special_cats])
        cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    # === 自定义日期范围选择 ===
    custom_dates_selected = False
    t1 = None
    t2 = None

    if "Custom dates" in time_range:
        custom_dates_selected = True
        st.markdown("#### 📅 Custom Date Range")
        col_from, col_to, _ = st.columns([1, 1, 1])
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

    # 检查是否有有效选择
    has_time_range = bool(time_range)
    has_data_sel = bool(data_sel)
    has_cats_sel = bool(cats_sel)

    # 对于 Custom dates，需要确保日期已选择
    if "Custom dates" in time_range:
        has_valid_custom_dates = (t1 is not None and t2 is not None)
    else:
        has_valid_custom_dates = True

    # 实时计算图表数据
    if has_time_range and has_data_sel and has_cats_sel and has_valid_custom_dates:
        with st.spinner("Generating chart..."):
            combined_df = prepare_chart_data_fast(
                daily, category_tx, inv_grouped, time_range, data_sel, cats_sel,
                custom_dates_selected, t1, t2
            )

        if combined_df is not None and not combined_df.empty:
            # 立即显示图表
            fig = px.line(
                combined_df,
                x="date",
                y="value",
                color="series",
                title="All Selected Data Types by Category",
                labels={"date": "Date", "value": "Value", "series": "Series"}
            )

            # 改为欧洲日期格式
            fig.update_layout(
                xaxis=dict(tickformat="%d/%m/%Y"),
                hovermode="x unified",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # 显示数据表格
            with st.expander("View combined data for all selected types"):
                display_df = combined_df.copy()
                display_df["date"] = display_df["date"].dt.strftime("%d/%m/%Y")  # 改为欧洲日期格式

                # 对表格中的 Daily Net Sales 也进行四舍五入取整
                display_df.loc[display_df["data_type"] == "Daily Net Sales", "value"] = display_df.loc[
                    display_df["data_type"] == "Daily Net Sales", "value"
                ].apply(lambda x: proper_round(x) if not pd.isna(x) else 0)

                display_df = display_df.rename(columns={
                    "date": "Date",
                    "Category": "Category",
                    "data_type": "Data Type",
                    "value": "Value"
                })
                display_df = display_df.sort_values(["Date", "Category", "Data Type"])
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No data available for the selected combination.")