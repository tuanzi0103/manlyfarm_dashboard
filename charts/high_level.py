import streamlit as st
import pandas as pd
import plotly.express as px
import math
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

    # 修正的 SQL 查询：处理字符串格式的Tax数据
    daily_sql = """
    WITH transaction_totals AS (
        SELECT 
            date(Datetime) AS date,
            [Transaction ID] AS txn_id,
            -- 计算每个 Transaction 的总 Gross Sales 和总 Tax
            SUM([Gross Sales]) AS total_gross_sales,
            -- 处理字符串格式的Tax数据：移除$符号并转换为数字
            SUM(COALESCE(CAST(REPLACE(REPLACE([Tax], '$', ''), ',', '') AS REAL), 0)) AS total_tax,
            SUM(Qty) AS total_qty
        FROM transactions
        GROUP BY date, [Transaction ID]
    )
    SELECT
        date,
        -- 按 Transaction 去重后的总和 (Gross Sales - Tax)
        SUM(total_gross_sales - total_tax) AS net_sales_with_tax,
        SUM(total_gross_sales) AS gross_sales,
        SUM(total_tax) AS total_tax,
        COUNT(DISTINCT txn_id) AS transactions,
        CASE 
            WHEN COUNT(DISTINCT txn_id) > 0 
            THEN SUM(total_gross_sales - total_tax) * 1.0 / COUNT(DISTINCT txn_id)
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
            -- 按 Transaction + Category 聚合
            SUM([Net Sales]) AS cat_net_sales,
            -- 处理字符串格式的Tax数据
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
            -- 每个 Transaction 在该类别下的总额 (Net Sales + Tax) - 保持bar类的原始逻辑
            SUM(cat_net_sales + cat_tax) AS cat_total_with_tax,
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

    # 新增：获取客户数量的 SQL 查询
    customers_sql = """
    WITH cleaned AS (
        SELECT 
            date(Datetime) AS date,
            [Transaction ID] AS txn_id,
            CASE 
                WHEN [Customer ID] IS NULL OR TRIM([Customer ID]) = '' 
                THEN 'anon_' || [Transaction ID] 
                ELSE TRIM([Customer ID]) 
            END AS customer_key
        FROM transactions
    ),
    unique_customers AS (
        SELECT 
            date,
            COUNT(DISTINCT customer_key) AS unique_customers
        FROM cleaned
        GROUP BY date
    )
    SELECT
        date,
        unique_customers
    FROM unique_customers
    ORDER BY date;
    """

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)
    customers_df = pd.read_sql(customers_sql, db)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])

    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])

    if not customers_df.empty:
        customers_df["date"] = pd.to_datetime(customers_df["date"])
        # 将客户数据合并到 daily 数据中
        daily = daily.merge(customers_df, on="date", how="left")
        daily["unique_customers"] = daily["unique_customers"].fillna(0)

    return daily, category


@st.cache_data
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

    # === 用 catalogue 现算 ===
    df["Quantity"] = pd.to_numeric(df.get("Current Quantity Vie Market & Bar", 0), errors="coerce").fillna(0).abs()
    df["Price"] = pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0)
    df["UnitCost"] = pd.to_numeric(df.get("Default Unit Cost", 0), errors="coerce").fillna(0)

    def calc_retail(row):
        O, AA, tax = row["Price"], row["Quantity"], str(row.get("Tax - GST (10%)", "")).strip().upper()
        return (O / 11 * 10) * AA if tax == "Y" else O * AA

    df["Retail Total"] = df.apply(calc_retail, axis=1)
    df["Inventory Value"] = df["UnitCost"] * df["Quantity"]
    df["Profit"] = df["Retail Total"] - df["Inventory Value"]

    # 聚合
    g = (
        df.groupby(["date", "Category"], as_index=False)[["Inventory Value", "Profit"]]
        .sum(min_count=1)
    )

    latest_date = g["date"].max() if not g.empty else None
    return g, latest_date


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    daily, category_tx = get_high_level_data()
    inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv)

    if daily.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    # === 特定日期选择 ===
    st.subheader("📅 Select Specific Date")
    available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
    selected_date = st.selectbox("Choose a specific date to view data", available_dates)

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    # 筛选选定日期的数据
    df_selected_date = daily[daily["date"] == selected_date_ts]

    today = pd.Timestamp.today().normalize()
    latest_date_tx = daily["date"].max()
    df_latest_tx = daily[daily["date"] == latest_date_tx]

    # === KPI（交易，口径按小票） ===
    # 使用选定日期的数据 - 确保使用 net_sales_with_tax (Gross Sales - Tax)
    kpis_main = {
        "Daily Net Sales": proper_round(df_selected_date["net_sales_with_tax"].sum()),
        "Daily Transactions": df_selected_date["transactions"].sum(),
        "Number of Customers": df_selected_date[
            "unique_customers"].sum() if "unique_customers" in df_selected_date.columns else 0,
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
                    # 对于Daily Transactions、Number of Customers和Items Sold，去掉前面的$符号
                    if label in ["Daily Transactions", "Number of Customers", "Items Sold"]:
                        display = f"{proper_round(val):,}"
                    # 对于Avg Transaction，保留两位小数
                    elif label == "Avg Transaction":
                        display = f"${val:.2f}"
                    else:
                        display = f"${proper_round(val):,}"
                with col:
                    st.markdown(f"<div style='font-size:28px; font-weight:600'>{display}</div>", unsafe_allow_html=True)
                    st.caption(label)
                    if label in captions:
                        st.caption(captions[label])

    st.markdown("---")

    # === 交互选择 ===
    st.subheader("🔍 Select Parameters")

    # 添加 CSS 来限制多选框高度（兼容 Streamlit 1.50）
    st.markdown("""
    <style>
    /* 控制 multiselect 下拉选项的最大显示高度（新版结构） */
    div[data-baseweb="popover"] ul {
        max-height: 6em !important;  /* 大约显示3条 */
        overflow-y: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Time range 选择 - 独立处理
    time_range_options = ["Custom dates", "WTD", "MTD", "YTD"]
    time_range = st.multiselect("Choose time range", time_range_options, key="hl_time")

    # 如果选择了 Custom dates，立即显示日期选择
    custom_dates_selected = False
    t1 = None
    t2 = None

    if "Custom dates" in time_range:
        custom_dates_selected = True
        col1, col2 = st.columns(2)
        with col1:
            t1 = st.date_input("From", value=today - pd.Timedelta(days=7))
        with col2:
            t2 = st.date_input("To", value=today)

    data_options = [
        "Daily Net Sales", "Daily Transactions", "Number of Customers", "Avg Transaction", "3M Avg", "6M Avg",
        "Inventory Value", "Profit (Amount)", "Items Sold"
    ]
    data_sel = persisting_multiselect("Choose data type", data_options, key="hl_data")

    # 修正bar分类名称 - 与数据库中的实际名称一致
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    if category_tx is None or category_tx.empty:
        st.info("No category breakdown available.")
        return

    all_cats_tx = sorted(category_tx["Category"].fillna("Unknown").unique().tolist())
    # 将bar和retail放在最前面
    all_cats_extended = ["bar", "retail"] + sorted(set(all_cats_tx) - {"bar", "retail"})
    cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    # 只要有time range选择就继续，不强制三个都有值
    if time_range and data_sel and cats_sel:
        grouped_tx = category_tx.copy()

        # 时间范围筛选
        if "WTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=7)]
        if "MTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=30)]
        if "YTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=365)]
        if custom_dates_selected and t1 and t2:
            grouped_tx = grouped_tx[
                (grouped_tx["date"] >= pd.to_datetime(t1)) & (grouped_tx["date"] <= pd.to_datetime(t2))]

        grouped_inv = inv_grouped.copy()
        if not grouped_inv.empty:
            if "WTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=7)]
            if "MTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=30)]
            if "YTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=365)]
            if custom_dates_selected and t1 and t2:
                grouped_inv = grouped_inv[
                    (grouped_inv["date"] >= pd.to_datetime(t1)) & (grouped_inv["date"] <= pd.to_datetime(t2))]

        small_cats = [c for c in cats_sel if c not in ("bar", "retail")]
        parts_tx = []

        if small_cats:
            parts_tx.append(grouped_tx[grouped_tx["Category"].isin(small_cats)])

        # === 完全分开的bar聚合逻辑 - 保持原始逻辑 (net sale + tax) ===
        if "bar" in cats_sel:
            # 直接从原始数据中筛选bar类别
            bar_categories_list = list(bar_cats)
            bar_df = category_tx[category_tx["Category"].isin(bar_categories_list)].copy()

            # 应用相同的时间范围筛选
            if "WTD" in time_range:
                bar_df = bar_df[bar_df["date"] >= today - pd.Timedelta(days=7)]
            if "MTD" in time_range:
                bar_df = bar_df[bar_df["date"] >= today - pd.Timedelta(days=30)]
            if "YTD" in time_range:
                bar_df = bar_df[bar_df["date"] >= today - pd.Timedelta(days=365)]
            if custom_dates_selected and t1 and t2:
                bar_df = bar_df[(bar_df["date"] >= pd.to_datetime(t1)) & (bar_df["date"] <= pd.to_datetime(t2))]

            if not bar_df.empty:
                # 按日期聚合bar数据 - 保持原始逻辑：使用包含tax的net_sales_with_tax (Net Sales + Tax)
                agg = (bar_df.groupby("date", as_index=False)
                       .agg(net_sales_with_tax=("net_sales_with_tax", "sum"),
                            net_sales=("net_sales", "sum"),
                            total_tax=("total_tax", "sum"),
                            transactions=("transactions", "sum"),
                            gross=("gross", "sum"),
                            qty=("qty", "sum")))
                agg["avg_txn"] = (agg["net_sales_with_tax"] / agg["transactions"]).replace([pd.NA, float("inf")], 0)
                agg["Category"] = "bar"
                parts_tx.append(agg)

        # === 完全分开的retail聚合逻辑 - 使用总Daily Net Sales减去Bar部分 ===
        if "retail" in cats_sel:
            # 获取选定时间范围内的总Daily Net Sales (Gross Sales - Tax)
            daily_filtered = daily.copy()

            # 应用相同的时间范围筛选到daily数据
            if "WTD" in time_range:
                daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=7)]
            if "MTD" in time_range:
                daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=30)]
            if "YTD" in time_range:
                daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=365)]
            if custom_dates_selected and t1 and t2:
                daily_filtered = daily_filtered[
                    (daily_filtered["date"] >= pd.to_datetime(t1)) & (daily_filtered["date"] <= pd.to_datetime(t2))]

            # 计算每天的bar总额 (保持原始逻辑：net sale + tax)
            bar_daily_totals = pd.DataFrame()
            if "bar" in cats_sel and not bar_df.empty:
                bar_daily_totals = (bar_df.groupby("date", as_index=False)
                                    .agg(bar_total=("net_sales_with_tax", "sum")))

            # 创建retail数据框
            retail_data = []
            for date_val in daily_filtered["date"].unique():
                # 总Daily Net Sales (Gross Sales - Tax)
                date_total = daily_filtered[daily_filtered["date"] == date_val]["net_sales_with_tax"].sum()

                # 查找该日期的bar总额 (net sale + tax)
                bar_total = 0
                if not bar_daily_totals.empty and date_val in bar_daily_totals["date"].values:
                    bar_total = bar_daily_totals[bar_daily_totals["date"] == date_val]["bar_total"].iloc[0]

                # Retail = 总Daily Net Sales - Bar总额
                retail_total = proper_round(date_total - bar_total)

                # 获取该日期的交易数和数量
                date_transactions = daily_filtered[daily_filtered["date"] == date_val]["transactions"].sum()
                date_qty = daily_filtered[daily_filtered["date"] == date_val]["qty"].sum()

                retail_data.append({
                    "date": date_val,
                    "net_sales_with_tax": retail_total,
                    "net_sales": retail_total,  # 对于retail，net_sales与net_sales_with_tax相同
                    "total_tax": 0,  # retail部分不包含tax
                    "transactions": date_transactions,
                    "avg_txn": retail_total / date_transactions if date_transactions > 0 else 0,
                    "gross": 0,  # retail部分不包含gross
                    "qty": date_qty,
                    "Category": "retail"
                })

            if retail_data:
                retail_agg = pd.DataFrame(retail_data)
                parts_tx.append(retail_agg)

        # === 计算每日总计 - 使用新的计算逻辑 ===
        if parts_tx:
            # 创建包含bar和retail的合并数据
            combined_tx = pd.concat(parts_tx, ignore_index=True)

            # 按日期计算bar和retail的总和
            daily_totals = combined_tx[combined_tx["Category"].isin(["bar", "retail"])].groupby("date",
                                                                                                as_index=False).agg(
                net_sales_with_tax=("net_sales_with_tax", "sum"),
                net_sales=("net_sales", "sum"),
                total_tax=("total_tax", "sum"),
                transactions=("transactions", "sum"),
                gross=("gross", "sum"),
                qty=("qty", "sum")
            )
            daily_totals["avg_txn"] = (daily_totals["net_sales_with_tax"] / daily_totals["transactions"]).replace(
                [pd.NA, float("inf")], 0)
            daily_totals["Category"] = "total"
            parts_tx.append(daily_totals)

        grouped_tx = pd.concat(parts_tx, ignore_index=True) if parts_tx else grouped_tx.iloc[0:0]

        parts_inv = []
        if not grouped_inv.empty:
            if small_cats:
                parts_inv.append(grouped_inv[grouped_inv["Category"].isin(small_cats)])

            if "bar" in cats_sel:
                bar_inv = grouped_inv[grouped_inv["Category"].isin(list(bar_cats))]
                if not bar_inv.empty:
                    agg = (bar_inv.groupby("date", as_index=False)
                           .agg(**{"Inventory Value": ("Inventory Value", "sum"),
                                   "Profit": ("Profit", "sum")}))
                    agg["Category"] = "bar"
                    parts_inv.append(agg)

            if "retail" in cats_sel:
                retail_inv = grouped_inv[~grouped_inv["Category"].isin(list(bar_cats))]
                if not retail_inv.empty:
                    agg = (retail_inv.groupby("date", as_index=False)
                           .agg(**{"Inventory Value": ("Inventory Value", "sum"),
                                   "Profit": ("Profit", "sum")}))
                    agg["Category"] = "retail"
                    parts_inv.append(agg)

        grouped_inv = pd.concat(parts_inv, ignore_index=True) if parts_inv else grouped_inv.iloc[0:0]

        mapping_tx = {
            "Daily Net Sales": ("net_sales_with_tax", "Daily Net Sales"),
            "Daily Transactions": ("transactions", "Daily Transactions"),
            "Number of Customers": ("transactions", "Number of Customers"),  # 注意：这里需要从daily数据获取
            "Avg Transaction": ("avg_txn", "Avg Transaction"),
            "3M Avg": ("net_sales_with_tax", "3M Avg (Rolling 90d)"),
            "6M Avg": ("net_sales_with_tax", "6M Avg (Rolling 180d)"),
            "Items Sold": ("qty", "Items Sold"),
        }
        mapping_inv = {
            "Inventory Value": ("Inventory Value", "Inventory Value"),
            "Profit (Amount)": ("Profit", "Profit (Retail - Inventory)"),
        }

        for metric in data_sel:
            if metric in mapping_tx:
                y, title = mapping_tx[metric]

                # 对于Number of Customers，需要从daily数据获取
                if metric == "Number of Customers":
                    daily_filtered = daily.copy()
                    # 应用相同的时间范围筛选
                    if "WTD" in time_range:
                        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=7)]
                    if "MTD" in time_range:
                        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=30)]
                    if "YTD" in time_range:
                        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=365)]
                    if custom_dates_selected and t1 and t2:
                        daily_filtered = daily_filtered[
                            (daily_filtered["date"] >= pd.to_datetime(t1)) & (
                                        daily_filtered["date"] <= pd.to_datetime(t2))]

                    # 创建客户数据图表
                    if "unique_customers" in daily_filtered.columns:
                        fig = px.line(daily_filtered, x="date", y="unique_customers", title=title, markers=True)
                        fig.update_layout(xaxis=dict(type="date"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No customer data available.")
                else:
                    # 在显示时过滤掉total类别
                    plot_df = grouped_tx[grouped_tx["Category"] != "total"].dropna(subset=[y]).copy()

                    if metric in ["3M Avg", "6M Avg"]:
                        if metric == "3M Avg":
                            plot_df["rolling"] = plot_df.groupby("Category")[y].transform(
                                lambda x: x.rolling(90, min_periods=1).mean())
                        else:
                            plot_df["rolling"] = plot_df.groupby("Category")[y].transform(
                                lambda x: x.rolling(180, min_periods=1).mean())
                        fig = px.line(plot_df, x="date", y="rolling", color="Category", title=title, markers=True)
                    else:
                        fig = px.line(plot_df, x="date", y=y, color="Category", title=title, markers=True)
                    fig.update_layout(xaxis=dict(type="date"))
                    st.plotly_chart(fig, use_container_width=True)

            elif metric in mapping_inv:
                y, title = mapping_inv[metric]
                if grouped_inv is not None and not grouped_inv.empty:
                    plot_df = grouped_inv.dropna(subset=[y]).copy()
                    fig = px.line(plot_df, x="date", y=y, color="Category", title=title, markers=True)
                    fig.update_layout(xaxis=dict(type="date"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No inventory data to plot for {metric}.")

        st.subheader("📋 Detailed Data")
        tables = []

        if not grouped_tx.empty:
            cols_tx = ["date", "Category"]
            for sel in data_sel:
                if sel in mapping_tx and sel != "Number of Customers":  # 排除Number of Customers
                    cols_tx.append(mapping_tx[sel][0])
            table_tx = grouped_tx[cols_tx].copy()

            # 在显示时过滤掉total类别
            table_tx = table_tx[table_tx["Category"] != "total"]

            # 使用标准的四舍五入方法
            for col in table_tx.columns:
                if col in ["net_sales_with_tax", "avg_txn", "net_sales"]:
                    table_tx[f"{col}_raw"] = table_tx[col]  # 保存原始值用于调试
                    table_tx[col] = table_tx[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
                elif col in ["transactions", "qty"]:
                    table_tx[col] = table_tx[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            table_tx["date"] = table_tx["date"].dt.strftime("%Y-%m-%d")

            tables.append(
                table_tx.drop(columns=[col for col in table_tx.columns if col.endswith('_raw')], errors='ignore'))

        if not grouped_inv.empty:
            cols_inv = ["date", "Category"]
            for sel in data_sel:
                if sel in mapping_inv:
                    cols_inv.append(mapping_inv[sel][0])
            table_inv = grouped_inv[cols_inv].copy()

            # 使用标准的四舍五入方法
            for col in table_inv.columns:
                if col in ["Inventory Value", "Profit"]:
                    table_inv[col] = table_inv[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            table_inv["date"] = table_inv["date"].dt.strftime("%Y-%m-%d")
            tables.append(table_inv)

        if tables:
            out = pd.concat(tables, ignore_index=True)
            st.dataframe(out, use_container_width=True)
        else:
            st.info("No data for the selected filters.")
    else:
        st.info("Please select time range, data, and category to generate the chart.")