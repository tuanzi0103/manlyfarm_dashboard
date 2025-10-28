import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from services.db import get_db


def proper_round(x):
    """标准的四舍五入方法，0.5总是向上舍入"""
    if pd.isna(x):
        return x
    return round(x)


def persisting_multiselect(label, options, key, default=None):
    """持久化多选框，处理默认值不在选项中的情况"""
    if key not in st.session_state:
        st.session_state[key] = default or []

    # 过滤掉不在当前选项中的默认值
    st.session_state[key] = [item for item in st.session_state[key] if item in options]

    return st.multiselect(label, options, default=st.session_state[key], key=key)


def persisting_multiselect_with_width(label, options, key, default=None, width_chars=None):
    """持久化多选框，带宽度控制（与 high_level.py 一致）"""
    if key not in st.session_state:
        st.session_state[key] = default or []

    # 过滤掉不在当前选项中的默认值
    st.session_state[key] = [item for item in st.session_state[key] if item in options]

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
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads", "Breakfast Bowls"}

    # 复制数据避免修改原数据
    df = df_filtered.copy()

    # === 修改：所有Bar分类也使用net_sales（不含税）===
    df["final_sales"] = df.apply(
        lambda row: row["net_sales"] if row["Category"] in bar_cats else row["net_sales"],
        axis=1
    )

    # === 修改：移除这里的四舍五入，在汇总后再进行 ===
    # 不再在数据准备阶段进行四舍五入

    return df


def extract_brand_name(item_name):
    """
    提取品牌：对清洗后的 Item 名称取第一个词作为品牌。
    这样像 "TLD Frenchs Forest Raw Honey 1Kg" -> "TLD"
          "HTG Organic Maple Syrup 1L" -> "HTG"
          "SPIRAL ORG Maple Syrup 250ml" -> "SPIRAL"
          "HANDHOE Macadamia Butter Roasted Crunchy 225g" -> "HANDHOE"
          "Beerose Honey 500g" -> "BEEROSE"
    避免把 'Butter/Honey/Maple/Jam/Tahini' 等产品词识别成品牌。
    """
    import re
    if pd.isna(item_name):
        return "Other"

    # 先用你已有的清洗函数做末尾规格/前缀清理
    cleaned = clean_item_name_for_comments(str(item_name))

    # 去掉多余空白
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Other"

    # 按空格或连字符等分割
    tokens = re.split(r"[ \t\-_/]+", cleaned)

    # 取第一个"看起来像品牌"的 token：
    # - 至少含字母
    # - 非纯数字
    for tok in tokens:
        has_alpha = any(c.isalpha() for c in tok)
        if has_alpha and not tok.isdigit():
            # 清掉结尾的逗号/点号之类
            tok = tok.strip(",.;:()[]{}")
            if tok:
                return tok.upper()

    return "Other"


def calculate_item_sales(items_df, selected_categories, selected_items, start_date=None, end_date=None):
    """计算指定category和items的销售数据"""
    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 如果有选中的分类，则应用分类筛选
    if selected_categories:
        filtered_items = filtered_items[filtered_items["Category"].isin(selected_categories)]

    # 清理商品名称用于匹配 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 如果有选中的商品，则应用商品项筛选
    if selected_items:
        filtered_items = filtered_items[filtered_items["clean_item"].isin(selected_items)]

    if filtered_items.empty:
        return pd.DataFrame()

    # 定义bar分类
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads", "Breakfast Bowls"}

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        # === 修改：所有Bar分类也使用Net Sales（不含税）===
        if row["Category"] in bar_cats:
            # Bar分类：现在只使用Net Sales（不含税）
            return row["Net Sales"]  # 不再在这里四舍五入
        else:
            # 非Bar分类：直接使用Net Sales
            return row["Net Sales"]  # 不再在这里四舍五入

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 按商品项汇总
    item_summary = filtered_items.groupby(["Category", "clean_item"]).agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # === 修改：在汇总后进行四舍五入 ===
    item_summary["Qty"] = item_summary["Qty"].apply(lambda x: int(proper_round(x)) if pd.notna(x) else 0)
    item_summary["final_sales"] = item_summary["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    return item_summary.rename(columns={
        "clean_item": "Item",
        "Qty": "Sum of Items Sold",
        "final_sales": "Sum of Daily Sales"
    })[["Category", "Item", "Sum of Items Sold", "Sum of Daily Sales"]]


def calculate_item_daily_trends(items_df, selected_categories, selected_items, start_date=None, end_date=None):
    """计算指定category和items的每日趋势数据"""
    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 如果有选中的分类，则应用分类筛选
    if selected_categories:
        filtered_items = filtered_items[filtered_items["Category"].isin(selected_categories)]

    # 清理商品名称用于匹配 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 如果有选中的商品，则应用商品项筛选
    if selected_items:
        filtered_items = filtered_items[filtered_items["clean_item"].isin(selected_items)]

    if filtered_items.empty:
        return pd.DataFrame()

    # 定义bar分类
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads", "Breakfast Bowls"}

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        # === 修改：所有Bar分类也使用Net Sales（不含税）===
        if row["Category"] in bar_cats:
            # Bar分类：现在只使用Net Sales（不含税）
            return row["Net Sales"]  # 不再在这里四舍五入
        else:
            # 非Bar分类：直接使用Net Sales
            return row["Net Sales"]  # 不再在这里四舍五入

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 按日期和商品项汇总
    daily_trends = filtered_items.groupby(["date", "Category", "clean_item"]).agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # 按日期汇总所有选中商品的总和
    daily_summary = daily_trends.groupby("date").agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # === 修改：在汇总后进行四舍五入 ===
    daily_summary["Qty"] = daily_summary["Qty"].apply(lambda x: int(proper_round(x)) if pd.notna(x) else 0)
    daily_summary["final_sales"] = daily_summary["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    return daily_summary.rename(columns={
        "Qty": "Sum of Items Sold",
        "final_sales": "Sum of Daily Sales"
    })[["date", "Sum of Items Sold", "Sum of Daily Sales"]]


def clean_item_name_for_comments(item):
    """清理商品名称 - 移除所有计量单位但保留商品名"""
    if pd.isna(item):
        return item

    # 移除所有类型的计量单位（重量、容量等）
    import re
    # 匹配数字后跟g/kg/ml/L/升/毫升/oz/lb等模式，移除整个计量单位部分
    pattern = r'\s*\d+\.?\d*\s*(g|kg|ml|mL|L|升|毫升|oz|lb)\s*$'
    cleaned = re.sub(pattern, '', str(item), flags=re.IGNORECASE)

    # 移除所有 "XXX - " 这种前缀模式（比如 "$460 WRAP -", "$360 BREAKFAST -", "$345 BURRITO -"）
    cleaned = re.sub(r'^.*?[a-zA-Z]+\s*-\s*', '', cleaned)

    # 移除首尾空格
    cleaned = cleaned.strip()

    return cleaned


def get_top_items_by_category(items_df, categories, start_date=None, end_date=None, for_total=False):
    """获取每个分类销量前3的商品，按品牌分组
    for_total: 如果为True，则返回整个分类组的前3品牌
    """
    if not categories:
        return {}

    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 过滤指定分类的商品
    filtered_items = filtered_items[filtered_items["Category"].isin(categories)]

    if filtered_items.empty:
        return {}

    # 定义bar分类
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads", "Breakfast Bowls"}

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        # === 修改：所有Bar分类也使用Net Sales（不含税）===
        if row["Category"] in bar_cats:
            # Bar分类：现在只使用Net Sales（不含税）
            return row["Net Sales"]  # 不再在这里四舍五入
        else:
            # 非Bar分类：直接使用Net Sales
            return row["Net Sales"]  # 不再在这里四舍五入

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 清理商品名称 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 提取品牌名称 - 使用改进的品牌检测
    filtered_items["brand"] = filtered_items["clean_item"].apply(extract_brand_name)

    if for_total:
        # 对于总计行，获取整个分类组的前3品牌
        brand_sales = filtered_items.groupby("brand").agg({
            "final_sales": "sum"
        }).reset_index()

        # === 修改：在汇总后进行四舍五入 ===
        brand_sales["final_sales"] = brand_sales["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        if not brand_sales.empty:
            top_3 = brand_sales.nlargest(3, "final_sales")
            # 格式：$销售额 品牌名
            top_brands_list = [f"${int(row['final_sales'])} {row['brand']}" for _, row in top_3.iterrows()]
            return ", ".join(top_brands_list)
        else:
            return "No items"
    else:
        # 对于普通行，获取每个分类的前3品牌
        category_brands = filtered_items.groupby(["Category", "brand"]).agg({
            "final_sales": "sum"
        }).reset_index()

        # === 修改：在汇总后进行四舍五入 ===
        category_brands["final_sales"] = category_brands["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # 获取每个分类的前3品牌
        top_brands_by_category = {}
        for category in categories:
            category_data = category_brands[category_brands["Category"] == category]
            if not category_data.empty:
                top_3 = category_data.nlargest(3, "final_sales")
                # 格式：$销售额 品牌名
                top_brands_list = [f"${int(row['final_sales'])} {row['brand']}" for _, row in top_3.iterrows()]
                top_brands_by_category[category] = ", ".join(top_brands_list)
            else:
                top_brands_by_category[category] = "No items"

        return top_brands_by_category


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    # === 全局样式: 让 st.dataframe 里的所有表格文字左对齐 ===
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style='font-size:22px; font-weight:700; margin-top:-2rem !important; margin-bottom:0.2rem !important;'>🧾 Sales Report by Category</h2>
    <style>
    /* 去掉 Streamlit 默认标题和上一个元素之间的间距 */
    div.block-container h2 {
        padding-top: 0 !important;
        margin-top: -2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 预加载所有数据 - 使用与high_level.py相同的数据源
    with st.spinner("Loading data..."):
        daily, category_tx, items_df = preload_all_data()

    if category_tx.empty:
        st.info("No category data available.")
        return

    # ---------------- Time Range Filter ----------------
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>📅 Time Range</h4>", unsafe_allow_html=True)

    # 🔹 使用三列布局缩短下拉框宽度，与 high_level.py 保持一致
    col1, col2, col3, _ = st.columns([1, 1, 1, 4])

    with col1:
        # 应用与 high_level.py 相同的选择框样式
        range_opt = st.selectbox("Select range", ["Custom dates", "WTD", "MTD", "YTD"], key="sr_range")

    today = pd.Timestamp.today().normalize()
    start_date, end_date = None, today

    if range_opt == "Custom dates":
        # ==== ✅ 自动计算最近有数据的一周 ====
        if not category_tx.empty:
            all_dates = sorted(category_tx["date"].dt.normalize().unique())
            today = pd.Timestamp.today().normalize()
            this_monday = today - pd.Timedelta(days=today.weekday())  # 当前周一
            this_sunday = this_monday + pd.Timedelta(days=6)

            # 当前周的日期范围
            this_week_mask = (category_tx["date"] >= this_monday) & (category_tx["date"] <= this_sunday)
            this_week_data = category_tx.loc[this_week_mask]

            if not this_week_data.empty:
                # ✅ 当前周有数据，默认显示当前周
                default_from, default_to = this_monday, min(this_sunday, all_dates[-1])
            else:
                # ✅ 当前周无数据，则回退上一周
                last_monday = this_monday - pd.Timedelta(days=7)
                last_sunday = this_sunday - pd.Timedelta(days=7)
                last_week_mask = (category_tx["date"] >= last_monday) & (category_tx["date"] <= last_sunday)
                last_week_data = category_tx.loc[last_week_mask]
                if not last_week_data.empty:
                    default_from, default_to = last_monday, last_sunday
                else:
                    # 如果两周都没数据，则取最近有数据的一周
                    latest_date = pd.to_datetime(all_dates[-1])
                    default_to = latest_date
                    default_from = latest_date - pd.Timedelta(days=6)
        else:
            # 数据为空时回退默认
            today = pd.Timestamp.today().normalize()
            default_from, default_to = today - pd.Timedelta(days=7), today

        # === 日期选择器 ===
        col_from, col_to, _ = st.columns([1, 1, 5])
        with col_from:
            t1 = st.date_input(
                "From",
                value=default_from,
                key="sr_date_from",
                format="DD/MM/YYYY"
            )
        with col_to:
            t2 = st.date_input(
                "To",
                value=default_to,
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
            # 只显示Top 10分类
            g_top10_items = g.head(10)
            fig1 = px.bar(g_top10_items, x="Category", y="items_sold", title="Items Sold (by Category) - Top 10",
                          height=400)
            fig1.update_layout(margin=dict(t=60, b=60))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            # 只显示Top 10分类
            g_sorted = g.sort_values("daily_sales", ascending=False).head(10)
            fig2 = px.bar(g_sorted, x="Category", y="daily_sales", title="Daily Sales (by Category) - Top 10",
                          height=400)
            fig2.update_layout(margin=dict(t=60, b=60))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data under current filters.")
        return

    # ---------------- Group definitions ----------------
    # 使用与 high_level.py 完全相同的分类定义
    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads", "Breakfast Bowls"}
    retail_cats = [c for c in df_filtered_fixed["Category"].unique() if c not in bar_cats]

    # helper: 根据时间范围计算汇总数据 - 使用修复后的数据
    def time_range_summary(data, cats, range_type, start_dt, end_dt):
        sub = data[data["Category"].isin(cats)].copy()
        if sub.empty:
            return pd.DataFrame()

        # 使用修复后的数据聚合 - 先不四舍五入
        summary = sub.groupby("Category", as_index=False).agg(
            items_sold=("qty", "sum"),
            daily_sales=("final_sales", "sum")  # 使用修复后的销售额
        )

        # === 修改：移除这里的四舍五入，在计算完所有汇总后再进行 ===
        # summary["items_sold"] = summary["items_sold"].apply(lambda x: proper_round(x) if pd.notna(x) else x)
        # summary["daily_sales"] = summary["daily_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # 计算与前一个相同长度时间段的对比
        if start_dt and end_dt:
            time_diff = end_dt - start_dt
            prev_start = start_dt - time_diff - timedelta(days=1)
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

        # === 修改：修正weekly_change计算逻辑 ===
        # 新逻辑：(当前周销售额 - 前一周销售额) / 前一周销售额 * 100
        MIN_BASE = 50
        summary["weekly_change"] = np.where(
            summary["prior_daily_sales"] > MIN_BASE,
            (summary["daily_sales"] - summary["prior_daily_sales"]) / summary["prior_daily_sales"] * 100,  # 乘以100
            np.nan
        )

        # 计算日均销量
        if start_dt and end_dt:
            days_count = (end_dt - start_dt).days + 1
            summary["per_day"] = summary["items_sold"] / days_count
        else:
            summary["per_day"] = summary["items_sold"] / 7  # 默认按7天计算

        # === 修改：保留原始 daily_sales 精度，用于 Total 汇总 ===
        summary["daily_sales_raw"] = summary["daily_sales"]  # 保存原始浮点值供后续计算

        # 仅 items_sold 取整
        summary["items_sold"] = summary["items_sold"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # 展示列用整数，但不影响 raw 精度
        summary["daily_sales_display"] = summary["daily_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # per_day 也取整展示
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
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>📊 Bar Categories</h4>", unsafe_allow_html=True)
    bar_df = time_range_summary(df_filtered_fixed, bar_cats, range_opt, start_date, end_date)

    if not bar_df.empty:
        # 获取Bar分类的前3品牌
        bar_top_items = get_top_items_by_category(items_df, bar_cats, start_date, end_date, for_total=False)
        # 获取Bar分类组的前3品牌（用于总计行）
        bar_total_top_items = get_top_items_by_category(items_df, bar_cats, start_date, end_date, for_total=True)

        # 添加Comments列
        bar_df["Comments"] = bar_df["Category"].map(bar_top_items)

        bar_df = bar_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales": "Sum of Daily Sales",
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })

        bar_df = bar_df.sort_values("Sum of Daily Sales", ascending=False)
        # 创建总计行
        total_items_sold = bar_df["Sum of Items Sold"].sum()
        total_daily_sales = bar_df["Sum of Daily Sales"].sum()
        total_per_day = bar_df["Per day"].sum()

        # 计算Total行的Weekly change - 基于总销售额与前一周期的对比
        total_prior_sales = bar_df["prior_daily_sales"].sum()
        MIN_BASE = 50
        if total_prior_sales > MIN_BASE:
            total_weekly_change = (total_daily_sales - total_prior_sales) / total_prior_sales * 100  # 乘以100
        else:
            total_weekly_change = np.nan

        # 创建数据框（与high_level.py相同的格式）- 总计行放在第一行
        bar_summary_data = {
            'Row Labels': ["Total"] + bar_df["Row Labels"].tolist(),
            'Sum of Items Sold': [total_items_sold] + bar_df["Sum of Items Sold"].tolist(),
            'Sum of Daily Sales': [total_daily_sales] + bar_df["Sum of Daily Sales"].tolist(),
            'Weekly change': [total_weekly_change] + bar_df["Weekly change"].tolist(),
            'Per day': [total_per_day] + bar_df["Per day"].tolist(),
            'Comments': [bar_total_top_items] + bar_df["Comments"].tolist()
        }

        df_bar_summary = pd.DataFrame(bar_summary_data)

        # === 修正：直接按照Weekly change数值从小到大排序 ===
        # 先分离Total行和其他行
        total_row = df_bar_summary[df_bar_summary['Row Labels'] == 'Total']
        other_rows = df_bar_summary[df_bar_summary['Row Labels'] != 'Total']

        # === 修正：按 Weekly change 数值从小到大排序（空值放最后）
        other_rows_sorted = other_rows.sort_values(
            by='Weekly change',
            key=lambda x: pd.to_numeric(x, errors='coerce'),
            ascending=True,
            na_position='last'
        )

        # Total 行始终放在最上方
        df_bar_summary_sorted = pd.concat([total_row, other_rows_sorted], ignore_index=True)

        # === ✅ 保持等宽且保留自定义列宽 ===
        TABLE_WIDTH = 730

        # === 定义统一列宽配置（保留原来比例） ===
        bar_column_config = {
            "Row Labels": st.column_config.Column(width=130),
            "Sum of Items Sold": st.column_config.Column(width=110),
            "Sum of Daily Sales": st.column_config.NumberColumn(width=130, format="$%d"),
            "Per day": st.column_config.Column(width=70),
            "Comments": st.column_config.Column(width=240),
            "Weekly change": st.column_config.NumberColumn(width=100, label="Weekly change", format="%.2f%%"),
        }

        # === 固定宽度的CSS，不改列宽比例，只统一外框 ===
        st.markdown(f"""
        <style>
        .bar-table-wrapper {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            min-width:{TABLE_WIDTH}px !important;
            overflow-x:hidden !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] table {{
            table-layout: fixed !important;
            width:{TABLE_WIDTH}px !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] td,
        .bar-table-wrapper [data-testid="stDataFrame"] th {{
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # === 两个表放在同一个容器 ===
        st.markdown(f"<div class='bar-table-wrapper' style='border: 0.1px solid #e6e6e6; padding: 0px; margin: 0px;'>",
                    unsafe_allow_html=True)

        # Total 表

        st.dataframe(
            total_row[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                       "Per day", "Comments", "Weekly change"]],
            column_config=bar_column_config,
            hide_index=True,
            use_container_width=False  # 🚫 不自动平分列宽
        )
        # === 添加：减少两个表格之间的间距 ===
        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            margin-top: -16px !important;
            margin-bottom: -16px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 主表
        st.dataframe(
            other_rows_sorted[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                               "Per day", "Comments", "Weekly change"]],
            column_config=bar_column_config,
            hide_index=True,
            use_container_width=False  # 🚫 不自动平分列宽
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Bar分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📦 Bar Category Items</h4>", unsafe_allow_html=True)

        # 获取所有Bar分类的商品项
        bar_items_df = items_df[items_df["Category"].isin(bar_cats)].copy()
        if not bar_items_df.empty:
            # 使用新的清理函数移除所有计量单位
            bar_items_df["clean_item"] = bar_items_df["Item"].apply(clean_item_name_for_comments)
            bar_item_options = sorted(bar_items_df["clean_item"].dropna().unique())

            # 选择Bar分类和商品项 - 放在同一行
            col_bar1, col_bar2, col_bar3, _ = st.columns([1.2, 1.6, 1.3, 2.9])
            with col_bar1:
                selected_bar_categories = persisting_multiselect_with_width(
                    "Select Bar Categories",
                    options=sorted(bar_df["Row Labels"].unique()),
                    key="bar_categories_select",
                    width_chars=22
                )
            with col_bar2:
                selected_bar_items = persisting_multiselect_with_width(
                    "Select Items from Bar Categories",
                    options=bar_item_options,
                    key="bar_items_select",
                    width_chars=30
                )

            # 显示选中的商品项数据
            if selected_bar_categories or selected_bar_items:
                bar_item_summary = calculate_item_sales(
                    items_df, selected_bar_categories, selected_bar_items, start_date, end_date
                )

                if not bar_item_summary.empty:
                    # 设置列配置
                    item_column_config = {
                        'Category': st.column_config.Column(width="150px"),
                        'Item': st.column_config.Column(width="200px"),
                        'Sum of Items Sold': st.column_config.Column(width="130px"),
                        'Sum of Daily Sales': st.column_config.Column(width="100px")
                    }

                    st.dataframe(bar_item_summary, column_config=item_column_config, use_container_width=False)

                    # 显示小计
                    total_qty = bar_item_summary["Sum of Items Sold"].sum()
                    total_sales = bar_item_summary["Sum of Daily Sales"].sum()
                    st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales}")

                    # 显示每日趋势折线图
                    bar_daily_trends = calculate_item_daily_trends(
                        items_df, selected_bar_categories, selected_bar_items, start_date, end_date
                    )

                    if not bar_daily_trends.empty:
                        # 创建折线图
                        fig = go.Figure()

                        # 添加Sum of Items Sold线
                        fig.add_trace(go.Scatter(
                            x=bar_daily_trends["date"],
                            y=bar_daily_trends["Sum of Items Sold"],
                            mode='lines+markers',
                            name='Sum of Items Sold',
                            line=dict(color='blue')
                        ))

                        # 添加Sum of Daily Sales线（使用次坐标轴）
                        fig.add_trace(go.Scatter(
                            x=bar_daily_trends["date"],
                            y=bar_daily_trends["Sum of Daily Sales"],
                            mode='lines+markers',
                            name='Sum of Daily Sales',
                            line=dict(color='red'),
                            yaxis='y2'
                        ))

                        # 设置图表布局
                        fig.update_layout(
                            title="Daily Trends for Selected Items",
                            xaxis_title="Date",
                            yaxis=dict(
                                title=dict(text="Sum of Items Sold", font=dict(color='blue')),
                                tickfont=dict(color='blue')
                            ),
                            yaxis2=dict(
                                title=dict(text="Sum of Daily Sales ($)", font=dict(color='red')),
                                tickfont=dict(color='red'),
                                overlaying='y',
                                side='right'
                            ),
                            height=400,
                            margin=dict(t=60, b=60)
                        )

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected items.")
        else:
            st.info("No items found in Bar categories.")

    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table ----------------
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>🛍️ Retail Categories</h4>", unsafe_allow_html=True)
    retail_df = time_range_summary(df_filtered_fixed, retail_cats, range_opt, start_date, end_date)

    if not retail_df.empty:
        # 获取Retail分类的前3品牌
        retail_top_items = get_top_items_by_category(items_df, retail_cats, start_date, end_date, for_total=False)
        # 获取Retail分类组的前3品牌（用于总计行）
        retail_total_top_items = get_top_items_by_category(items_df, retail_cats, start_date, end_date, for_total=True)

        # 添加Comments列
        retail_df["Comments"] = retail_df["Category"].map(retail_top_items)

        retail_df = retail_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales_display": "Sum of Daily Sales",  # ✅ 改为用取整展示列
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })

        retail_df = retail_df.sort_values("Sum of Daily Sales", ascending=False)

        # 创建总计行
        # === 修复：先用原始浮点数计算百分比，再四舍五入显示 ===
        total_daily_sales_raw = retail_df["daily_sales_raw"].sum()
        total_prior_sales_raw = retail_df["prior_daily_sales"].sum()
        MIN_BASE = 50
        if total_prior_sales_raw > MIN_BASE:
            total_weekly_change = (total_daily_sales_raw - total_prior_sales_raw) / total_prior_sales_raw * 100  # 乘以100
        else:
            total_weekly_change = np.nan

        # 显示时再四舍五入
        total_items_sold = proper_round(retail_df["Sum of Items Sold"].sum())
        total_daily_sales = proper_round(total_daily_sales_raw)
        total_per_day = proper_round(retail_df["Per day"].sum())

        # 创建数据框（与high_level.py相同的格式）- 总计行放在第一行
        retail_summary_data = {
            'Row Labels': ["Total"] + retail_df["Row Labels"].tolist(),
            'Sum of Items Sold': [total_items_sold] + retail_df["Sum of Items Sold"].tolist(),
            'Sum of Daily Sales': [total_daily_sales] + retail_df["Sum of Daily Sales"].tolist(),
            'Weekly change': [total_weekly_change] + retail_df["Weekly change"].tolist(),
            'Per day': [total_per_day] + retail_df["Per day"].tolist(),
            'Comments': [retail_total_top_items] + retail_df["Comments"].tolist()
        }

        df_retail_summary = pd.DataFrame(retail_summary_data)

        # === 修正：直接按照Weekly change数值从小到大排序 ===
        # 先分离Total行和其他行
        total_row = df_retail_summary[df_retail_summary['Row Labels'] == 'Total']
        other_rows = df_retail_summary[df_retail_summary['Row Labels'] != 'Total']

        # === 修正：按 Weekly change 数值从小到大排序（空值放最后）
        other_rows_sorted = other_rows.sort_values(
            by='Weekly change',
            key=lambda x: pd.to_numeric(x, errors='coerce'),
            ascending=True,
            na_position='last'
        )

        # Total 行始终放在最上方
        df_retail_summary_sorted = pd.concat([total_row, other_rows_sorted], ignore_index=True)

        # === ✅ Retail Category: Total单独列出 + 灰线 + 保持列宽一致 ===
        TABLE_WIDTH = 730  # 跟Bar保持一致

        # === 拆分 Total 与其他行 ===
        total_row_retail = df_retail_summary_sorted[df_retail_summary_sorted['Row Labels'] == 'Total']
        other_rows_retail = df_retail_summary_sorted[df_retail_summary_sorted['Row Labels'] != 'Total']

        # === 定义统一列宽配置（与Bar一致） ===
        retail_column_config = {
            "Row Labels": st.column_config.Column(width=130),
            "Sum of Items Sold": st.column_config.Column(width=110),
            "Sum of Daily Sales": st.column_config.NumberColumn(width=130, format="$%d"),
            "Per day": st.column_config.Column(width=70),
            "Comments": st.column_config.Column(width=240),
            "Weekly change": st.column_config.NumberColumn(width=100, label="Weekly change", format="%.2f%%"),
        }

        # === CSS：强制两表等宽 ===
        st.markdown(f"""
        <style>
        .retail-table-wrapper {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            margin: 0;
            padding: 0;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            min-width:{TABLE_WIDTH}px !important;
            overflow-x:hidden !important;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] table {{
            table-layout: fixed !important;
            width:{TABLE_WIDTH}px !important;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] td,
        .retail-table-wrapper [data-testid="stDataFrame"] th {{
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # === 两表放同一容器，锁定一致宽度 ===
        with st.container():
            st.markdown("<div class='retail-table-wrapper'>", unsafe_allow_html=True)

            # --- Total表 ---
            st.dataframe(
                total_row_retail[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                                  "Per day", "Comments", "Weekly change"]],
                column_config=retail_column_config,
                hide_index=True,
                use_container_width=False
            )

            # === 添加：减少两个表格之间的间距 ===
            st.markdown("""
                    <style>
                    [data-testid="stDataFrame"] {
                        margin-top: -16px !important;
                        margin-bottom: -16px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

            # --- 主表 ---
            st.dataframe(
                other_rows_retail[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                                   "Per day", "Comments", "Weekly change"]],
                column_config=retail_column_config,
                hide_index=True,
                use_container_width=False
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # Retail分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📦 Retail Category Items</h4>", unsafe_allow_html=True)

        # 获取所有Retail分类的商品项
        retail_items_df = items_df[items_df["Category"].isin(retail_cats)].copy()
        if not retail_items_df.empty:
            # 使用新的清理函数移除所有计量单位
            retail_items_df["clean_item"] = retail_items_df["Item"].apply(clean_item_name_for_comments)
            retail_item_options = sorted(retail_items_df["clean_item"].dropna().unique())

            # 选择Retail分类和商品项 - 放在同一行
            col_retail1, col_retail2, col_retail3, _ = st.columns([1.2, 1.6, 1.3, 2.9])
            with col_retail1:
                selected_retail_categories = persisting_multiselect_with_width(
                    "Select Retail Categories",
                    options=sorted(retail_df["Row Labels"].unique()),
                    key="retail_categories_select",
                    width_chars=22
                )
            with col_retail2:
                selected_retail_items = persisting_multiselect_with_width(
                    "Select Items from Retail Categories",
                    options=retail_item_options,
                    key="retail_items_select",
                    width_chars=30
                )

            # 显示选中的商品项数据
            if selected_retail_categories or selected_retail_items:
                retail_item_summary = calculate_item_sales(
                    items_df, selected_retail_categories, selected_retail_items, start_date, end_date
                )

                if not retail_item_summary.empty:
                    # 设置列配置
                    item_column_config = {
                        'Category': st.column_config.Column(width="150px"),
                        'Item': st.column_config.Column(width="200px"),
                        'Sum of Items Sold': st.column_config.Column(width="130px"),
                        'Sum of Daily Sales': st.column_config.Column(width="100px")
                    }

                    st.dataframe(retail_item_summary, column_config=item_column_config, use_container_width=False)

                    # 显示小计
                    total_qty = retail_item_summary["Sum of Items Sold"].sum()
                    total_sales = retail_item_summary["Sum of Daily Sales"].sum()
                    st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales}")

                    # 显示每日趋势折线图
                    retail_daily_trends = calculate_item_daily_trends(
                        items_df, selected_retail_categories, selected_retail_items, start_date, end_date
                    )

                    if not retail_daily_trends.empty:
                        # 创建折线图
                        fig = go.Figure()

                        # 添加Sum of Items Sold线
                        fig.add_trace(go.Scatter(
                            x=retail_daily_trends["date"],
                            y=retail_daily_trends["Sum of Items Sold"],
                            mode='lines+markers',
                            name='Sum of Items Sold',
                            line=dict(color='blue')
                        ))

                        # 添加Sum of Daily Sales线（使用次坐标轴）
                        fig.add_trace(go.Scatter(
                            x=retail_daily_trends["date"],
                            y=retail_daily_trends["Sum of Daily Sales"],
                            mode='lines+markers',
                            name='Sum of Daily Sales',
                            line=dict(color='red'),
                            yaxis='y2'
                        ))

                        fig.update_layout(
                            title="Daily Trends for Selected Items",
                            xaxis_title="Date",
                            yaxis=dict(
                                title=dict(text="Sum of Items Sold", font=dict(color='blue')),
                                tickfont=dict(color='blue')
                            ),
                            yaxis2=dict(
                                title=dict(text="Sum of Daily Sales ($)", font=dict(color='red')),
                                tickfont=dict(color='red'),
                                overlaying='y',
                                side='right'
                            ),
                            height=400,
                            margin=dict(t=60, b=60)
                        )

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected items.")
        else:
            st.info("No items found in Retail categories.")

    else:
        st.info("No data for Retail categories.")