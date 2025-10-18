import streamlit as st
import pandas as pd
import plotly.express as px
import math
import numpy as np
from services.db import get_db
# === 添加页面配置 - 修复布局不一致问题 ===
st.set_page_config(
    page_title="Vie Manly Dashboard",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

# === 全局样式：消除顶部标题间距 ===
st.markdown("""
<style>
/* 去掉 Vie Manly Dashboard 与 High Level Report 之间的空白 */
div.block-container h1, 
div.block-container h2, 
div.block-container h3, 
div.block-container p {
    margin-top: 0rem !important;
    margin-bottom: 0rem !important;
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

/* 更强力地压缩 Streamlit 自动插入的 vertical space */
div.block-container > div {
    margin-top: 0rem !important;
    margin-bottom: 0rem !important;
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

/* 消除标题和选择框之间空隙 */
div[data-testid="stVerticalBlock"] > div {
    margin-top: 0rem !important;
    margin-bottom: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

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


def persisting_multiselect(label, options, key, default=None, width_chars=None):
    if key not in st.session_state:
        st.session_state[key] = default or []

    # === 修改：添加自定义宽度参数 ===
    if width_chars is None:
        # 默认宽度为标签长度+1字符
        label_width = len(label)
        min_width = label_width + 1
    else:
        # 使用自定义宽度
        min_width = width_chars

    st.markdown(f"""
    <style>
        /* 强制设置多选框宽度 */
        [data-testid*="{key}"] {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] > div {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] [data-baseweb="select"] {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] [data-baseweb="select"] > div {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    return st.multiselect(label, options, default=st.session_state[key], key=key)


# === 预加载所有数据 ===


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

        # 为每个分类计算滚动平均值
        category_with_rolling = []
        for cat in category["Category"].unique():
            cat_data = category[category["Category"] == cat].copy()
            # 按日期排序确保滚动计算正确
            cat_data = cat_data.sort_values("date")
            # 计算该分类的滚动平均值
            cat_data["3M_Avg_Rolling"] = cat_data["net_sales_with_tax"].rolling(window=90, min_periods=1,
                                                                                center=False).mean()
            cat_data["6M_Avg_Rolling"] = cat_data["net_sales_with_tax"].rolling(window=180, min_periods=1,
                                                                                center=False).mean()
            category_with_rolling.append(cat_data)

        # 重新组合数据
        category = pd.concat(category_with_rolling, ignore_index=True)

    return daily, category


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

    # 定义bar分类（这5个分类使用 net_sales + tax 计算）
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    # 修复：过滤掉没有数据的分类，避免重复显示
    small_cats = []
    for c in cats_sel:
        if c not in ("bar", "retail", "total"):
            small_cats.append(c)

    parts_tx = []

    if small_cats:
        # 为小类数据添加滚动平均值
        small_cats_data = grouped_tx[grouped_tx["Category"].isin(small_cats)].copy()

        # 修复：按日期和分类重新计算 net_sales_with_tax
        for cat in small_cats:
            cat_mask = small_cats_data["Category"] == cat
            if cat not in bar_cats:  # 非bar分类使用 net_sales 列
                # 按日期分组计算每个日期的 net_sales 总和
                daily_net_sales = small_cats_data[cat_mask].groupby("date")["net_sales"].sum().reset_index()
                # 结果四舍五入保留整数
                daily_net_sales["net_sales_with_tax"] = daily_net_sales["net_sales"].apply(
                    lambda x: proper_round(x) if not pd.isna(x) else 0
                )

                # 更新原始数据中的 net_sales_with_tax
                for _, row in daily_net_sales.iterrows():
                    date_mask = (small_cats_data["date"] == row["date"]) & (small_cats_data["Category"] == cat)
                    small_cats_data.loc[date_mask, "net_sales_with_tax"] = row["net_sales_with_tax"]

        parts_tx.append(small_cats_data)

    # 处理bar分类 - 重新计算bar的滚动平均
    if "bar" in cats_sel:
        bar_tx = grouped_tx[grouped_tx["Category"].isin(bar_cats)].copy()
        if not bar_tx.empty:
            # 先按日期聚合bar数据
            bar_daily_agg = bar_tx.groupby("date").agg({
                "net_sales_with_tax": "sum",
                "transactions": "sum",
                "qty": "sum",
                "3M_Avg_Rolling": "mean",  # 保留原有的滚动平均值
                "6M_Avg_Rolling": "mean"  # 保留原有的滚动平均值
            }).reset_index()

            # 计算bar的平均交易额
            bar_daily_agg["avg_txn"] = bar_daily_agg.apply(
                lambda x: x["net_sales_with_tax"] / x["transactions"] if x["transactions"] > 0 else 0,
                axis=1
            )

            # 为bar数据计算准确的滚动平均（如果需要重新计算）
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
            "qty": "total_qty",
            "3M_Avg_Rolling": "total_3M_Avg",
            "6M_Avg_Rolling": "total_6M_Avg"
        })

        # 获取每日bar数据
        bar_daily = grouped_tx[grouped_tx["Category"].isin(bar_cats)].groupby("date").agg({
            "net_sales_with_tax": "sum",
            "transactions": "sum",
            "qty": "sum",
            "3M_Avg_Rolling": "mean",
            "6M_Avg_Rolling": "mean"
        }).reset_index()
        bar_daily = bar_daily.rename(columns={
            "net_sales_with_tax": "bar_net_sales",
            "transactions": "bar_transactions",
            "qty": "bar_qty",
            "3M_Avg_Rolling": "bar_3M_Avg",
            "6M_Avg_Rolling": "bar_6M_Avg"
        })

        # 合并total和bar数据
        retail_data = total_daily.merge(bar_daily, on="date", how="left")

        # 计算retail = total - bar
        retail_data["net_sales_with_tax"] = retail_data["total_net_sales"] - retail_data["bar_net_sales"].fillna(0)
        retail_data["transactions"] = retail_data["total_transactions"] - retail_data["bar_transactions"].fillna(0)
        retail_data["qty"] = retail_data["total_qty"] - retail_data["bar_qty"].fillna(0)

        # 计算retail的滚动平均值
        retail_data["3M_Avg_Rolling"] = retail_data["net_sales_with_tax"].rolling(window=90, min_periods=1,
                                                                                  center=False).mean()
        retail_data["6M_Avg_Rolling"] = retail_data["net_sales_with_tax"].rolling(window=180, min_periods=1,
                                                                                  center=False).mean()

        # 计算平均交易额
        retail_data["avg_txn"] = retail_data.apply(
            lambda x: x["net_sales_with_tax"] / x["transactions"] if x["transactions"] > 0 else 0,
            axis=1
        )

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

    # 数据映射 - 修改2：为每个数据类型都添加3M和6M Avg的映射
    data_map_extended = {
        "Daily Net Sales": "net_sales_with_tax",
        "Daily Transactions": "transactions",
        "Avg Transaction": "avg_txn",
        "Items Sold": "qty",
        "Inventory Value": "inventory_value",
        "Profit (Amount)": "profit_amount",
        # 为每个数据类型添加对应的3M和6M Avg
        "Daily Net Sales 3M Avg": "3M_Avg_Rolling",
        "Daily Net Sales 6M Avg": "6M_Avg_Rolling",
        "Daily Transactions 3M Avg": "transactions_3M_Avg",
        "Daily Transactions 6M Avg": "transactions_6M_Avg",
        "Avg Transaction 3M Avg": "avg_txn_3M_Avg",
        "Avg Transaction 6M Avg": "avg_txn_6M_Avg",
        "Items Sold 3M Avg": "qty_3M_Avg",
        "Items Sold 6M Avg": "qty_6M_Avg",
    }

    # 为其他数据类型计算3M和6M滚动平均值
    if any("3M Avg" in data_type or "6M Avg" in data_type for data_type in data_sel):
        # 为transactions计算滚动平均
        df_plot["transactions_3M_Avg"] = df_plot.groupby("Category")["transactions"].transform(
            lambda x: x.rolling(window=90, min_periods=1, center=False).mean()
        )
        df_plot["transactions_6M_Avg"] = df_plot.groupby("Category")["transactions"].transform(
            lambda x: x.rolling(window=180, min_periods=1, center=False).mean()
        )

        # 为avg_txn计算滚动平均
        df_plot["avg_txn_3M_Avg"] = df_plot.groupby("Category")["avg_txn"].transform(
            lambda x: x.rolling(window=90, min_periods=1, center=False).mean()
        )
        df_plot["avg_txn_6M_Avg"] = df_plot.groupby("Category")["avg_txn"].transform(
            lambda x: x.rolling(window=180, min_periods=1, center=False).mean()
        )

        # 为qty计算滚动平均
        df_plot["qty_3M_Avg"] = df_plot.groupby("Category")["qty"].transform(
            lambda x: x.rolling(window=90, min_periods=1, center=False).mean()
        )
        df_plot["qty_6M_Avg"] = df_plot.groupby("Category")["qty"].transform(
            lambda x: x.rolling(window=180, min_periods=1, center=False).mean()
        )

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

        # 修复：确保日期按正确顺序排序
        combined_df = combined_df.sort_values("date")

        return combined_df

    return None


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    # === 全局样式：消除顶部标题间距 ===
    st.markdown("""
    <style>
    /* 去掉 Vie Manly Dashboard 与 High Level Report 之间的空白 */
    div.block-container h1, 
    div.block-container h2, 
    div.block-container h3, 
    div.block-container p {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 更强力地压缩 Streamlit 自动插入的 vertical space */
    div.block-container > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 消除标题和选择框之间空隙 */
    div[data-testid="stVerticalBlock"] > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === 保留标题 ===
    st.markdown("<h2 style='font-size:24px; font-weight:700;'>📊 High Level Report</h2>", unsafe_allow_html=True)

    # 在现有的样式后面添加：
    st.markdown("""
    <style>
    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    div[data-baseweb="select"] {
        min-width: 12ch !important;
        max-width: 20ch !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 预加载所有数据
    with st.spinner("Loading data..."):
        daily, category_tx = preload_all_data()
        inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv)

    if daily.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    # === 特定日期选择 ===
    col_date, _ = st.columns([1, 2])
    with col_date:
        available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
        # 将日期格式改为欧洲格式显示
        available_dates_formatted = [date.strftime('%d/%m/%Y') for date in available_dates]

        # === 修改2：日期选择框宽度精确匹配日期长度 ===
        # 计算最长日期的长度（欧洲格式 dd/mm/yyyy = 10字符）
        date_width = 18  # dd/mm/yyyy 固定10字符
        selectbox_width = date_width + 1  # 加1给下拉箭头

        st.markdown(f"""
        <style>
            /* 日期选择框容器 - 精确宽度 */
            div[data-testid*="stSelectbox"] {{
                width: {selectbox_width}ch !important;
                min-width: {selectbox_width}ch !important;
                max-width: {selectbox_width}ch !important;
                display: inline-block !important;
            }}
            /* 日期选择框标签 */
            div[data-testid*="stSelectbox"] label {{
                white-space: nowrap !important;
                font-size: 0.9rem !important;
                width: 100% !important;
            }}
            /* 下拉菜单 */
            div[data-testid*="stSelectbox"] [data-baseweb="select"] {{
                width: {selectbox_width}ch !important;
                min-width: {selectbox_width}ch !important;
                max-width: {selectbox_width}ch !important;
            }}
            /* 下拉选项容器 */
            div[role="listbox"] {{
                min-width: {selectbox_width}ch !important;
                max-width: {selectbox_width}ch !important;
            }}
            /* 隐藏多余的下拉箭头空间 */
            div[data-testid*="stSelectbox"] [data-baseweb="select"] > div {{
                padding-right: 0 !important;
            }}
        </style>
        """, unsafe_allow_html=True)

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

    # === 计算bar和retail的特定日期数据 ===
    def calculate_bar_retail_data(category_tx, selected_date, daily_data):
        """计算bar和retail在选定日期的数据"""
        selected_date_ts = pd.Timestamp(selected_date)

        # bar分类定义
        bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}

        # 筛选选定日期的分类数据
        daily_category_data = category_tx[category_tx["date"].dt.date == selected_date]

        # 计算bar数据
        bar_data = daily_category_data[daily_category_data["Category"].isin(bar_cats)]
        bar_net_sales = proper_round(bar_data["net_sales_with_tax"].sum())
        bar_transactions = bar_data["transactions"].sum()
        bar_avg_txn = bar_net_sales / bar_transactions if bar_transactions > 0 else 0
        bar_qty = bar_data["qty"].sum()

        # 计算bar的3M和6M平均值（使用最近的滚动平均值）
        bar_3m_avg = proper_round(bar_data["3M_Avg_Rolling"].iloc[-1]) if not bar_data.empty else 0
        bar_6m_avg = proper_round(bar_data["6M_Avg_Rolling"].iloc[-1]) if not bar_data.empty else 0

        # 计算retail数据 = total - bar
        total_data = daily_data[daily_data["date"].dt.date == selected_date]
        total_net_sales = proper_round(total_data["net_sales_with_tax"].sum())
        total_transactions = total_data["transactions"].sum()
        total_qty = total_data["qty"].sum()

        retail_net_sales = total_net_sales - bar_net_sales
        retail_transactions = total_transactions - bar_transactions
        retail_avg_txn = retail_net_sales / retail_transactions if retail_transactions > 0 else 0
        retail_qty = total_qty - bar_qty

        # 计算retail的3M和6M平均值（使用最近的滚动平均值）
        retail_3m_avg = proper_round(total_data["3M_Avg_Rolling"].iloc[-1]) - bar_3m_avg if not total_data.empty else 0
        retail_6m_avg = proper_round(total_data["6M_Avg_Rolling"].iloc[-1]) - bar_6m_avg if not total_data.empty else 0

        # 计算bar和retail的客户数量（这里简化处理，按交易比例分配）
        total_customers = calculate_customer_count(tx, selected_date)
        bar_customers = int(total_customers * (bar_transactions / total_transactions)) if total_transactions > 0 else 0
        retail_customers = total_customers - bar_customers

        return {
            "bar": {
                "Daily Net Sales": bar_net_sales,
                "Daily Transactions": bar_transactions,
                "# of Customers": bar_customers,
                "Avg Transaction": bar_avg_txn,
                "3M Avg": bar_3m_avg,
                "6M Avg": bar_6m_avg,
                "Items Sold": bar_qty
            },
            "retail": {
                "Daily Net Sales": retail_net_sales,
                "Daily Transactions": retail_transactions,
                "# of Customers": retail_customers,
                "Avg Transaction": retail_avg_txn,
                "3M Avg": retail_3m_avg,
                "6M Avg": retail_6m_avg,
                "Items Sold": retail_qty
            }
        }

    # === KPI（交易，口径按小票） ===
    kpis_main = {
        "Daily Net Sales": proper_round(df_selected_date["net_sales_with_tax"].sum()),
        "Daily Transactions": df_selected_date["transactions"].sum(),
        "# of Customers": calculate_customer_count(tx, selected_date),
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

    # 计算bar和retail数据
    bar_retail_data = calculate_bar_retail_data(category_tx, selected_date, daily)

    # 显示选定日期（字体加大）
    st.markdown(
        f"<h3 style='font-size:18px; font-weight:700;'>Selected Date: {selected_date.strftime('%d/%m/%Y')}</h3>",
        unsafe_allow_html=True)

    # ===== 组装三行数据 =====
    total_row = [
        f"${proper_round(kpis_main['Daily Net Sales']):,}",
        f"{proper_round(kpis_main['Daily Transactions']):,}",
        f"{proper_round(kpis_main['# of Customers']):,}",
        f"${kpis_main['Avg Transaction']:.2f}",
        f"${proper_round(kpis_main['3M Avg']):,}",
        f"${proper_round(kpis_main['6M Avg']):,}",
        f"{proper_round(kpis_main['Items Sold']):,}",
        f"${proper_round(inv_value_latest):,} <br><span style='font-size:10px; color:#666;'>as of {pd.to_datetime(inv_latest_date).strftime('%d/%m/%Y') if inv_latest_date else '-'}</span>"
    ]

    bar_row = [
        f"${proper_round(bar_retail_data['bar']['Daily Net Sales']):,}",
        f"{proper_round(bar_retail_data['bar']['Daily Transactions']):,}",
        f"{proper_round(bar_retail_data['bar']['# of Customers']):,}",
        f"${bar_retail_data['bar']['Avg Transaction']:.2f}",
        f"${proper_round(bar_retail_data['bar']['3M Avg']):,}",
        f"${proper_round(bar_retail_data['bar']['6M Avg']):,}",
        f"{proper_round(bar_retail_data['bar']['Items Sold']):,}",
        "-"
    ]

    retail_row = [
        f"${proper_round(bar_retail_data['retail']['Daily Net Sales']):,}",
        f"{proper_round(bar_retail_data['retail']['Daily Transactions']):,}",
        f"{proper_round(bar_retail_data['retail']['# of Customers']):,}",
        f"${bar_retail_data['retail']['Avg Transaction']:.2f}",
        f"${proper_round(bar_retail_data['retail']['3M Avg']):,}",
        f"${proper_round(bar_retail_data['retail']['6M Avg']):,}",
        f"{proper_round(bar_retail_data['retail']['Items Sold']):,}",
        "-"
    ]

    # ===== 渲染成 HTML 表格 =====
    # === 新增：Summary Table列宽配置 ===
    column_widths = {
        "label": "110px",
        "Daily Net Sales": "130px",
        "Daily Transactions": "140px",
        "# of Customers": "140px",
        "Avg Transaction": "125px",
        "3M Avg": "115px",
        "6M Avg": "115px",
        "Items Sold": "115px",
        "Inventory Value": "140px"
    }

    # 创建数据框
    summary_data = {
        '': ['Bar', 'Retail', 'Total'],
        'Daily Net Sales': [
            f"${proper_round(bar_retail_data['bar']['Daily Net Sales']):,}",
            f"${proper_round(bar_retail_data['retail']['Daily Net Sales']):,}",
            f"${proper_round(kpis_main['Daily Net Sales']):,}"
        ],
        'Daily Transactions': [
            f"{proper_round(bar_retail_data['bar']['Daily Transactions']):,}",
            f"{proper_round(bar_retail_data['retail']['Daily Transactions']):,}",
            f"{proper_round(kpis_main['Daily Transactions']):,}"
        ],
        '# of Customers': [
            f"{proper_round(bar_retail_data['bar']['# of Customers']):,}",
            f"{proper_round(bar_retail_data['retail']['# of Customers']):,}",
            f"{proper_round(kpis_main['# of Customers']):,}"
        ],
        'Avg Transaction': [
            f"${bar_retail_data['bar']['Avg Transaction']:.2f}",
            f"${bar_retail_data['retail']['Avg Transaction']:.2f}",
            f"${kpis_main['Avg Transaction']:.2f}"
        ],
        '3M Avg': [
            f"${proper_round(bar_retail_data['bar']['3M Avg']):,}",
            f"${proper_round(bar_retail_data['retail']['3M Avg']):,}",
            f"${proper_round(kpis_main['3M Avg']):,}"
        ],
        '6M Avg': [
            f"${proper_round(bar_retail_data['bar']['6M Avg']):,}",
            f"${proper_round(bar_retail_data['retail']['6M Avg']):,}",
            f"${proper_round(kpis_main['6M Avg']):,}"
        ],
        'Items Sold': [
            f"{proper_round(bar_retail_data['bar']['Items Sold']):,}",
            f"{proper_round(bar_retail_data['retail']['Items Sold']):,}",
            f"{proper_round(kpis_main['Items Sold']):,}"
        ],
        'Inventory Value': [
            "-", "-",
            f"${proper_round(inv_value_latest):,} (as of {pd.to_datetime(inv_latest_date).strftime('%d/%m/%Y') if inv_latest_date else '-'})"
        ]

    }

    df_summary = pd.DataFrame(summary_data)

    # 设置列配置
    column_config = {
        '': st.column_config.Column(width=80),
        'Daily Net Sales': st.column_config.Column(width=100),
        'Daily Transactions': st.column_config.Column(width=120),
        '# of Customers': st.column_config.Column(width=100),
        'Avg Transaction': st.column_config.Column(width=105),
        '3M Avg': st.column_config.Column(width=55),
        '6M Avg': st.column_config.Column(width=55),
        'Items Sold': st.column_config.Column(width=75),
        'Inventory Value': st.column_config.Column(width=105),
    }
    # 显示表格
    st.markdown("<h4 style='font-size:16px; font-weight:700; margin-top:1rem;'>Summary Table</h4>",
                unsafe_allow_html=True)
    st.dataframe(
        df_summary,
        column_config=column_config,
        hide_index=True,
        use_container_width=False
    )

    st.markdown("---")

    # === 交互选择 ===
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>🔍 Select Parameters</h4>", unsafe_allow_html=True)

    # 分类选择
    if category_tx is None or category_tx.empty:
        st.info("No category breakdown available.")
        return

    # 过滤掉没有数据的分类 - 修复重复显示问题
    category_tx["Category"] = category_tx["Category"].astype(str).str.strip()
    all_cats_tx = (
        category_tx["Category"]
        .fillna("Unknown")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # 只保留有实际数据的分类
    valid_cats = []
    seen_cats = set()
    for cat in all_cats_tx:
        if cat not in seen_cats:
            seen_cats.add(cat)
            cat_data = category_tx[category_tx["Category"] == cat]
            if not cat_data.empty and cat_data["net_sales_with_tax"].sum() > 0:
                valid_cats.append(cat)

    special_cats = ["bar", "retail", "total"]
    all_cats_extended = special_cats + sorted([c for c in valid_cats if c not in special_cats])

    # === 四个多选框一行显示（使用 columns，等宽且靠左） ===

    # 定义每个框的宽度比例
    col1, col2, col3, col4, _ = st.columns([1.0, 1.2, 0.7, 1.5, 2.6])

    with col1:
        time_range = persisting_multiselect(
            "Choose time range",
            ["Custom dates", "WTD", "MTD", "YTD"],
            key="hl_time",
            width_chars=15
        )

    with col2:
        data_sel_base = persisting_multiselect(
            "Choose data types",
            ["Daily Net Sales", "Daily Transactions", "Avg Transaction", "Items Sold", "Inventory Value"],
            key="hl_data_base",
            width_chars=22
        )

    with col3:
        data_sel_avg = persisting_multiselect(
            "Choose averages",
            ["3M Avg", "6M Avg"],
            key="hl_data_avg",
            width_chars=6
        )

    with col4:
        cats_sel = persisting_multiselect(
            "Choose categories",
            all_cats_extended,
            key="hl_cats",
            width_chars=30
        )

    # 加一小段 CSS，让四个框左对齐、间距最小
    st.markdown("""
    <style>
    div[data-testid="column"] {
        padding: 0 4px !important;
    }
    div[data-baseweb="select"] {
        min-width: 5ch !important;
        max-width: 35ch !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 合并数据类型选择
    data_sel = data_sel_base.copy()

    # 如果选择了平均值，为每个选择的基础数据类型添加对应的平均值
    for avg_type in data_sel_avg:
        for base_type in data_sel_base:
            if base_type in ["Daily Net Sales", "Daily Transactions", "Avg Transaction", "Items Sold"]:
                combined_type = f"{base_type} {avg_type}"
                data_sel.append(combined_type)

    # 如果没有选择任何基础数据类型但有平均值，默认使用Daily Net Sales
    if not data_sel_base and data_sel_avg:
        for avg_type in data_sel_avg:
            data_sel.append(f"Daily Net Sales {avg_type}")

    # === 自定义日期范围选择 ===
    custom_dates_selected = False
    t1 = None
    t2 = None

    # === 📅 Custom Date Range（保持原逻辑 + 显示 dd/mm/yyyy 格式） ===
    if "Custom dates" in time_range:
        custom_dates_selected = True

        # 标题风格与 Select Specific Date 一致
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📅 Custom Date Range</h4>", unsafe_allow_html=True)

        # 列布局：与上面多选框等宽比例
        col_from, col_to, _ = st.columns([1, 1, 5])

        # 日期输入框 - 修改为 dd/mm/yy 格式
        with col_from:
            t1 = st.date_input(
                "From",
                value=pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
                key="date_from",
                format="DD/MM/YYYY"  # 修改这里
            )

        with col_to:
            t2 = st.date_input(
                "To",
                value=pd.Timestamp.today().normalize(),
                key="date_to",
                format="DD/MM/YYYY"  # 修改这里
            )

        # 移除原有的JavaScript格式化代码，因为现在使用内置format参数

    # 修改1：检查三个多选框是否都有选择
    has_time_range = bool(time_range)
    has_data_sel = bool(data_sel)
    has_cats_sel = bool(cats_sel)

    # 对于 Custom dates，需要确保日期已选择
    if "Custom dates" in time_range:
        has_valid_custom_dates = (t1 is not None and t2 is not None)
    else:
        has_valid_custom_dates = True

    # 实时计算图表数据 - 修改1：只有三个多选框都选择了才展示
    if has_time_range and has_data_sel and has_cats_sel and has_valid_custom_dates:
        with st.spinner("Generating chart..."):
            combined_df = prepare_chart_data_fast(
                daily, category_tx, inv_grouped, time_range, data_sel, cats_sel,
                custom_dates_selected, t1, t2
            )

        if combined_df is not None and not combined_df.empty:
            # 修复：确保图表中的日期按正确顺序显示
            combined_df = combined_df.sort_values("date")

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
            st.markdown("""
            <style>
            div[data-testid="stExpander"] > div:first-child {
                width: fit-content !important;
                max-width: 95% !important;
            }
            div[data-testid="stDataFrame"] {
                width: fit-content !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # 显示数据表格 - 直接展示，去掉下拉框
            st.markdown("#### 📊 Combined Data for All Selected Types")
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
            # 修复：按日期正确排序
            display_df["Date_dt"] = pd.to_datetime(display_df["Date"], format='%d/%m/%Y')
            display_df = display_df.sort_values(["Date_dt", "Category", "Data Type"])
            display_df = display_df.drop("Date_dt", axis=1)

            # === 修改1：表格容器宽度跟随表格内容 ===
            # 计算表格总宽度
            total_width = 0
            for column in display_df.columns:
                header_len = len(str(column))
                # 估算列宽：标题长度+数据最大长度+2字符边距
                data_len = display_df[column].astype(str).str.len().max()
                col_width = max(header_len, data_len) + 2
                total_width += col_width

            # 设置表格容器样式
            st.markdown(f"""
            <style>
            /* 表格容器 - 宽度跟随内容 */
            [data-testid="stExpander"] {{
                width: auto !important;
                min-width: {total_width}ch !important;
                max-width: 100% !important;
            }}
            /* 让表格左右可滚动 */
            [data-testid="stDataFrame"] div[role="grid"] {{
                overflow-x: auto !important;
                width: auto !important;
            }}
            /* 自动列宽，不强制占满 */
            [data-testid="stDataFrame"] table {{
                table-layout: auto !important;
                width: auto !important;
            }}
            /* 所有单元格左对齐 */
            [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
                text-align: left !important;
                justify-content: flex-start !important;
            }}
            /* 防止省略号 */
            [data-testid="stDataFrame"] td {{
                white-space: nowrap !important;
            }}
            </style>
            """, unsafe_allow_html=True)

            # === 新逻辑：列宽根据标题字符串长度设置 ===
            column_config = {}
            for column in display_df.columns:
                header_len = len(str(column))
                column_config[column] = st.column_config.Column(
                    column,
                    width=f"{header_len + 2}ch"
                )

            # 对3M/6M平均值列四舍五入保留两位小数
            avg_mask = display_df["Data Type"].str.contains("3M Avg|6M Avg", case=False, na=False)
            display_df.loc[avg_mask, "Value"] = display_df.loc[avg_mask, "Value"].apply(
                lambda x: round(x, 2) if pd.notna(x) else x
            )

            st.dataframe(display_df, use_container_width=False, column_config=column_config)

        else:
            st.warning("No data available for the selected combination.")
