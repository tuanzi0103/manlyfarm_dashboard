import streamlit as st
import pandas as pd
import plotly.express as px
import math
import hashlib
import time
from services.db import get_db

# ==================== 数据版本控制 ====================
def get_data_version():
    """获取数据版本，用于强制刷新缓存"""
    return st.session_state.get('data_version', 0)


def increment_data_version():
    """增加数据版本号，强制刷新所有缓存"""
    current = st.session_state.get('data_version', 0)
    st.session_state.data_version = current + 1


def show_refresh_indicator():
    """显示刷新指示器"""
    st.markdown('<div class="refresh-indicator" id="refreshIndicator"></div>', unsafe_allow_html=True)
    # 0.5秒后移除指示器
    st.markdown("""
    <script>
    setTimeout(function() {
        var indicator = document.getElementById('refreshIndicator');
        if (indicator) indicator.remove();
    }, 500);
    </script>
    """, unsafe_allow_html=True)


def clear_all_cache():
    """清除所有缓存 - 全局函数，用于所有模块"""
    # 清除session state中的缓存数据
    keys_to_clear = [
        'precomputed_data', 'data_loaded',
        'hl_time', 'hl_data', 'hl_cats',
        'last_data_hash', 'cached_filtered_data'
    ]

    for key in list(st.session_state.keys()):
        if any(cache_key in key for cache_key in keys_to_clear):
            del st.session_state[key]

    # 清除streamlit缓存
    try:
        get_high_level_data.clear()
        _prepare_inventory_grouped.clear()
        compute_filtered_data.clear()
    except:
        pass

    # 增加数据版本号
    increment_data_version()

    # 显示刷新指示器
    show_refresh_indicator()


# ==================== 数据哈希检测 ====================
def get_data_hash(tx, mem, inv):
    """生成数据哈希来检测数据变化"""
    hash_parts = []

    # 对每个数据框生成哈希
    for df, name in [(tx, 'tx'), (mem, 'mem'), (inv, 'inv')]:
        if df is not None and not df.empty:
            # 使用数据形状和内容的哈希
            try:
                shape_hash = hash((df.shape[0], df.shape[1]))
                content_hash = pd.util.hash_pandas_object(df).sum()
                hash_parts.append(f"{name}_{shape_hash}_{content_hash}")
            except:
                hash_parts.append(f"{name}_error")
        else:
            hash_parts.append(f"{name}_empty")

    combined_hash = "_".join(hash_parts)
    return hashlib.md5(combined_hash.encode()).hexdigest()


# ==================== 原有工具函数 ====================
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
    """
    一个持久化的 multiselect 控件：
    - 第一次创建时会用 default 初始化；
    - 后续运行时如果 session_state 中已有值，则不再传 default（防止冲突警告）。
    """

    # 如果 Session State 里已经存在值，则直接返回控件，不再传 default，避免警告
    if key in st.session_state:
        return st.multiselect(label, options, key=key)

    # 如果还没有初始化，先写入默认值
    init_value = default or []
    st.session_state[key] = init_value

    # 第一次创建控件时传入 default
    return st.multiselect(label, options, default=init_value, key=key)


# ==================== 数据获取函数（带版本控制） ====================
@st.cache_data(ttl=3600)
def get_high_level_data(_data_version):
    """
    添加数据版本参数，当版本变化时缓存自动失效
    注意：此处代码保持不变，只修改控件布局部分
    """
    # ... 保持原有代码不变 ...
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

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])

    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])

    return daily, category


@st.cache_data(ttl=3600)
def _prepare_inventory_grouped(inv: pd.DataFrame, _data_version):
    # ... 保持原有代码不变 ...
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


# ==================== 计算缓存函数（带版本控制） ====================
@st.cache_data(ttl=600)
def compute_filtered_data(time_range, data_sel, cats_sel, daily, category_tx, inv_grouped, today, _data_version):
    """缓存过滤和计算的结果"""
    # ... 保持原有代码不变 ...
    # 首先获取时间筛选后的daily数据
    daily_filtered = daily.copy()

    # 应用时间范围筛选到daily数据
    if "WTD" in time_range:
        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=7)]
    if "MTD" in time_range:
        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=30)]
    if "YTD" in time_range:
        daily_filtered = daily_filtered[daily_filtered["date"] >= today - pd.Timedelta(days=365)]

    grouped_tx = category_tx.copy()

    # 应用相同的时间范围筛选到grouped_tx
    if "WTD" in time_range:
        grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=7)]
    if "MTD" in time_range:
        grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=30)]
    if "YTD" in time_range:
        grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=365)]

    grouped_inv = inv_grouped.copy()
    if not grouped_inv.empty:
        if "WTD" in time_range:
            grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=7)]
        if "MTD" in time_range:
            grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=30)]
        if "YTD" in time_range:
            grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=365)]

    bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}
    small_cats = [c for c in cats_sel if c not in ("bar", "retail", "total")]
    parts_tx = []

    if small_cats:
        parts_tx.append(grouped_tx[grouped_tx["Category"].isin(small_cats)])

    # 计算bar总额
    bar_categories_list = list(bar_cats)
    bar_df = grouped_tx[grouped_tx["Category"].isin(bar_categories_list)].copy()

    bar_agg = pd.DataFrame()
    if not bar_df.empty:
        # 按日期聚合bar数据
        bar_agg = (bar_df.groupby("date", as_index=False)
                   .agg(net_sales_with_tax=("net_sales_with_tax", "sum"),
                        net_sales=("net_sales", "sum"),
                        total_tax=("total_tax", "sum"),
                        transactions=("transactions", "sum"),
                        gross=("gross", "sum"),
                        qty=("qty", "sum")))
        bar_agg["avg_txn"] = (bar_agg["net_sales_with_tax"] / bar_agg["transactions"]).replace(
            [pd.NA, float("inf")], 0)
        bar_agg["Category"] = "bar"

        if "bar" in cats_sel:
            parts_tx.append(bar_agg)

    # 计算retail
    if "retail" in cats_sel:
        bar_daily_totals = pd.DataFrame()
        if not bar_agg.empty:
            bar_daily_totals = bar_agg[["date", "net_sales_with_tax"]].rename(
                columns={"net_sales_with_tax": "bar_total"})

        retail_data = []
        for date_val in daily_filtered["date"].unique():
            date_total = daily_filtered[daily_filtered["date"] == date_val]["net_sales_with_tax"].sum()

            bar_total = 0
            if not bar_daily_totals.empty and date_val in bar_daily_totals["date"].values:
                bar_total = bar_daily_totals[bar_daily_totals["date"] == date_val]["bar_total"].iloc[0]

            retail_total = proper_round(date_total - bar_total)

            date_transactions = daily_filtered[daily_filtered["date"] == date_val]["transactions"].sum()
            date_qty = daily_filtered[daily_filtered["date"] == date_val]["qty"].sum()

            retail_data.append({
                "date": date_val,
                "net_sales_with_tax": retail_total,
                "net_sales": retail_total,
                "total_tax": 0,
                "transactions": date_transactions,
                "avg_txn": retail_total / date_transactions if date_transactions > 0 else 0,
                "gross": 0,
                "qty": date_qty,
                "Category": "retail"
            })

        if retail_data:
            retail_agg = pd.DataFrame(retail_data)
            parts_tx.append(retail_agg)

    # 计算total
    if "total" in cats_sel:
        total_data = []
        for date_val in daily_filtered["date"].unique():
            date_total = daily_filtered[daily_filtered["date"] == date_val]["net_sales_with_tax"].sum()

            date_transactions = daily_filtered[daily_filtered["date"] == date_val]["transactions"].sum()
            date_qty = daily_filtered[daily_filtered["date"] == date_val]["qty"].sum()

            total_data.append({
                "date": date_val,
                "net_sales_with_tax": date_total,
                "net_sales": date_total,
                "total_tax": daily_filtered[daily_filtered["date"] == date_val]["total_tax"].sum(),
                "transactions": date_transactions,
                "avg_txn": date_total / date_transactions if date_transactions > 0 else 0,
                "gross": daily_filtered[daily_filtered["date"] == date_val]["gross_sales"].sum(),
                "qty": date_qty,
                "Category": "total"
            })

        if total_data:
            total_agg = pd.DataFrame(total_data)
            parts_tx.append(total_agg)

    # 合并所有数据
    if parts_tx:
        grouped_tx = pd.concat(parts_tx, ignore_index=True)
        grouped_tx = grouped_tx.sort_values(["Category", "date"])
    else:
        grouped_tx = grouped_tx.iloc[0:0]

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

        if "total" in cats_sel:
            total_inv = grouped_inv.copy()
            if not total_inv.empty:
                agg = (total_inv.groupby("date", as_index=False)
                       .agg(**{"Inventory Value": ("Inventory Value", "sum"),
                               "Profit": ("Profit", "sum")}))
                agg["Category"] = "total"
                parts_inv.append(agg)

    grouped_inv = pd.concat(parts_inv, ignore_index=True) if parts_inv else grouped_inv.iloc[0:0]

    return grouped_tx, grouped_inv


# ==================== 优化的模块化函数 ====================

def render_cache_control():
    """渲染缓存控制组件"""
    with st.sidebar.expander("🔄 Cache Control"):
        if st.button("🔄 Refresh Data Cache", type="primary", use_container_width=True):
            clear_all_cache()
            st.success("✅ Data cache refreshed! New data will be loaded.")
            st.rerun()


def render_kpi_section(daily, tx, selected_date, inv_grouped, inv_latest_date):
    """渲染KPI指标部分"""
    st.markdown(f"### 📅 Selected Date: {selected_date}")

    # 计算客户数量
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

    selected_date_ts = pd.Timestamp(selected_date)
    df_selected_date = daily[daily["date"] == selected_date_ts]

    # 计算KPI指标
    kpis_main = {
        "Daily Net Sales": proper_round(df_selected_date["net_sales_with_tax"].sum()),
        "Daily Transactions": df_selected_date["transactions"].sum(),
        "Number of Customers": calculate_customer_count(tx, selected_date),
        "Avg Transaction": df_selected_date["avg_txn"].mean(),
        "3M Avg": proper_round(daily["net_sales_with_tax"].rolling(90, min_periods=1).mean().iloc[-1]),
        "6M Avg": proper_round(daily["net_sales_with_tax"].rolling(180, min_periods=1).mean().iloc[-1]),
        "Items Sold": df_selected_date["qty"].sum(),
    }

    inv_value_latest = 0.0
    profit_latest = 0.0
    if inv_grouped is not None and not inv_grouped.empty and inv_latest_date is not None:
        sub = inv_grouped[inv_grouped["date"] == inv_latest_date]
        inv_value_latest = float(pd.to_numeric(sub["Inventory Value"], errors="coerce").sum())
        profit_latest = float(pd.to_numeric(sub["Profit"], errors="coerce").sum())

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
                if pd.isna(val):
                    display = "-"
                else:
                    if label == "Avg Transaction":
                        display = f"{val:,.2f}"
                    elif label in ["Daily Transactions", "Items Sold", "Number of Customers"]:
                        display = f"{int(proper_round(val)):,}"
                    else:
                        display = f"${proper_round(val):,}"
                with col:
                    st.markdown(
                        f"<div style='font-size:28px; font-weight:600'>{display}</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(label)
                    if label in captions:
                        st.caption(captions[label])


def render_interactive_section(daily, category_tx, inv_grouped, data_version):
    """渲染交互式图表和过滤部分"""
    st.subheader("🔍 Select Parameters")

    # 使用紧凑的列布局，让控件更短
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        time_range_options = ["Custom dates", "WTD", "MTD", "YTD"]
        time_range = persisting_multiselect(
            "Choose time range",
            time_range_options,
            "hl_time"
        )

    with col2:
        data_options = [
            "Daily Net Sales", "Daily Transactions", "Avg Transaction", "3M Avg", "6M Avg",
            "Inventory Value", "Profit (Amount)", "Items Sold"
        ]
        data_sel = persisting_multiselect(
            "Choose data type",
            data_options,
            "hl_data"
        )

    with col3:
        bar_cats = {"Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"}
        all_cats_tx = sorted(category_tx["Category"].fillna("Unknown").unique().tolist())
        special_cats = ["bar", "retail", "total"]
        all_cats_extended = special_cats + sorted([c for c in all_cats_tx if c not in special_cats])
        cats_sel = persisting_multiselect(
            "Choose categories",
            all_cats_extended,
            "hl_cats"
        )

    custom_dates_selected = False
    t1 = None
    t2 = None

    # 自定义日期选择器使用与上面相同的三列布局
    if "Custom dates" in time_range:
        custom_dates_selected = True
        # 使用相同的三列布局来保持宽度一致
        date_col1, date_col2, date_col3 = st.columns([1, 1, 1])

        with date_col1:
            st.markdown("**From:**")
            t1 = st.date_input(
                "From Date",
                value=pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
                key="date_from",
                label_visibility="collapsed"
            )

        with date_col2:
            st.markdown("**To:**")
            t2 = st.date_input(
                "To Date",
                value=pd.Timestamp.today().normalize(),
                key="date_to",
                label_visibility="collapsed"
            )

        # 第三列留空以保持布局对齐
        with date_col3:
            st.write("")  # 空列用于对齐

    # 创建过滤参数的哈希键
    filter_params = {
        'time_range': tuple(time_range) if time_range else (),
        'data_sel': tuple(data_sel) if data_sel else (),
        'cats_sel': tuple(cats_sel) if cats_sel else (),
        't1': t1,
        't2': t2,
        'data_version': data_version
    }
    filter_hash = hashlib.md5(str(filter_params).encode()).hexdigest()
    cache_key_filtered = f'cached_filtered_data_{filter_hash}'

    # 检查是否需要重新计算
    current_filter_state = {
        'time_range': time_range,
        'data_sel': data_sel,
        'cats_sel': cats_sel,
        't1': t1,
        't2': t2
    }

    # 获取上一次的过滤状态
    last_filter_state = st.session_state.get('last_filter_state', {})

    # 只有当过滤条件确实发生变化时才重新计算
    filter_changed = current_filter_state != last_filter_state

    if time_range and data_sel and cats_sel:
        # 检查是否有缓存的过滤结果，或者过滤条件没有变化
        if cache_key_filtered in st.session_state and not filter_changed:
            grouped_tx, grouped_inv = st.session_state[cache_key_filtered]
        else:
            # 只有当过滤条件变化时才重新计算
            if filter_changed:
                with st.spinner("🔄 Processing data..."):
                    today = pd.Timestamp.today().normalize()
                    grouped_tx, grouped_inv = compute_filtered_data(
                        time_range, data_sel, cats_sel, daily, category_tx, inv_grouped, today, data_version
                    )

                    # 处理自定义日期范围
                    if custom_dates_selected and t1 and t2:
                        grouped_tx = grouped_tx[
                            (grouped_tx["date"] >= pd.to_datetime(t1)) & (grouped_tx["date"] <= pd.to_datetime(t2))]
                        if not grouped_inv.empty:
                            grouped_inv = grouped_inv[
                                (grouped_inv["date"] >= pd.to_datetime(t1)) & (
                                        grouped_inv["date"] <= pd.to_datetime(t2))]

                    # 缓存结果
                    st.session_state[cache_key_filtered] = (grouped_tx, grouped_inv)
                    # 更新上一次的过滤状态
                    st.session_state['last_filter_state'] = current_filter_state
            else:
                # 使用现有的缓存数据
                grouped_tx, grouped_inv = st.session_state[cache_key_filtered]

        mapping_tx = {
            "Daily Net Sales": ("net_sales_with_tax", "Daily Net Sales"),
            "Daily Transactions": ("transactions", "Daily Transactions"),
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
                plot_df = grouped_tx.dropna(subset=[y]).copy()
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
                if sel in mapping_tx:
                    cols_tx.append(mapping_tx[sel][0])
            table_tx = grouped_tx[cols_tx].copy()
            for col in table_tx.columns:
                if col in ["net_sales_with_tax", "avg_txn", "net_sales"]:
                    table_tx[f"{col}_raw"] = table_tx[col]
                    table_tx[col] = table_tx[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
                elif col in ["transactions", "qty"]:
                    table_tx[col] = table_tx[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            table_tx["date"] = table_tx["date"].dt.strftime("%Y-%m-%d")
            table_tx = table_tx.sort_values(["Category", "date"])
            tables.append(
                table_tx.drop(columns=[col for col in table_tx.columns if col.endswith('_raw')], errors='ignore'))

        if not grouped_inv.empty:
            cols_inv = ["date", "Category"]
            for sel in data_sel:
                if sel in mapping_inv:
                    cols_inv.append(mapping_inv[sel][0])
            table_inv = grouped_inv[cols_inv].copy()
            for col in table_inv.columns:
                if col in ["Inventory Value", "Profit"]:
                    table_inv[col] = table_inv[col].apply(lambda x: proper_round(x) if pd.notna(x) else x)
            table_inv["date"] = table_inv["date"].dt.strftime("%Y-%m-%d")
            table_inv = table_inv.sort_values(["Category", "date"])
            tables.append(table_inv)

        if tables:
            out = pd.concat(tables, ignore_index=True)
            st.dataframe(out, use_container_width=True)
        else:
            st.info("No data for the selected filters.")
    else:
        st.info("Please select time range, data, and category to generate the chart.")

# ==================== 优化的主函数 ====================
def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame, data_updated=False):
    st.header("📊 High Level Report")

    # 渲染缓存控制
    render_cache_control()

    # 检测数据变化
    current_data_hash = get_data_hash(tx, mem, inv)
    last_data_hash = st.session_state.get('last_data_hash')

    # 获取数据版本
    data_version = get_data_version()

    # 如果数据发生变化或者还没有加载数据，重新获取数据
    if (current_data_hash != last_data_hash or
            'precomputed_data' not in st.session_state or
            data_updated):

        with st.spinner("🔄 Loading data..."):
            daily, category_tx = get_high_level_data(data_version)
            inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv, data_version)

            # 缓存预处理的数据
            st.session_state.precomputed_data = {
                'daily': daily,
                'category_tx': category_tx,
                'inv_grouped': inv_grouped,
                'inv_latest_date': inv_latest_date
            }

            # 更新数据哈希
            st.session_state.last_data_hash = current_data_hash
    else:
        # 使用缓存的数据
        precomputed = st.session_state.precomputed_data
        daily = precomputed['daily']
        category_tx = precomputed['category_tx']
        inv_grouped = precomputed['inv_grouped']
        inv_latest_date = precomputed['inv_latest_date']

    # 日期选择器 - 使用紧凑布局
    date_col1, date_col2 = st.columns([1, 3])
    with date_col1:
        selected_date = st.date_input(
            "Select Date",
            value=min(pd.Timestamp.today().normalize().date(), daily["date"].max().date()),
            min_value=daily["date"].min().date(),
            max_value=pd.Timestamp.today().normalize().date(),
            key="date_selector"
        )

    # 渲染KPI部分
    render_kpi_section(daily, tx, selected_date, inv_grouped, inv_latest_date)

    # 渲染交互式部分
    render_interactive_section(daily, category_tx, inv_grouped, data_version)