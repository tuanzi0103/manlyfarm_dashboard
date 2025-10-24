import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional

from services.analytics import (
    forecast_top_consumers,
    sku_consumption_timeseries,
)
from services.simulator import simulate_consumption, simulate_consumption_timeseries


def detect_store_current_qty_col(df_inv: pd.DataFrame) -> Optional[str]:
    if df_inv is None or df_inv.empty:
        return None
    norm = {c: str(c).lower().strip() for c in df_inv.columns}
    for c, n in norm.items():
        if n.startswith("current quantity"):
            return c
    return None


def persisting_multiselect(label, options, key, default=None, width_chars=None):
    """
    保持选择状态的多选框函数 - 统一宽度和箭头显示（增强版）
    """
    if key not in st.session_state:
        st.session_state[key] = default or []

    if width_chars is None:
        min_width = 30  # 全局默认 30ch
    else:
        min_width = width_chars

    st.markdown(f"""
    <style>
    /* === 强制覆盖 stMultiSelect 宽度（仅限当前 key） === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"],
    [data-testid*="{key}"][data-testid="stMultiSelect"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        flex: 0 0 {min_width}ch !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉框主体 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"],
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"] > div {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 输入框 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] input {{
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉菜单 === */
    div[role="listbox"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        box-sizing: border-box !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    return st.multiselect(label, options, default=st.session_state[key], key=key)


def filter_by_time_range(df, time_range, custom_dates_selected=False, t1=None, t2=None):
    """根据时间范围筛选数据"""
    if df is None or df.empty:
        return df

    # 如果没有日期列，直接返回原数据
    if "date" not in df.columns and "source_date" not in df.columns:
        return df

    # 获取日期列名
    date_col = "date" if "date" in df.columns else "source_date"

    # 确保日期列是datetime类型
    df_filtered = df.copy()
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors="coerce")

    # 获取当前日期
    today = pd.Timestamp.today().normalize()

    # 计算时间范围
    start_of_week = today - pd.Timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    start_of_year = today.replace(month=1, day=1)

    # 应用时间范围筛选 - 这里要使用 date_col 变量而不是硬编码的 "date"
    if "WTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_week]
    if "MTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_month]
    if "YTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_year]
    if custom_dates_selected and t1 and t2:
        t1_ts = pd.to_datetime(t1)
        t2_ts = pd.to_datetime(t2)
        df_filtered = df_filtered[
            (df_filtered[date_col] >= t1_ts) & (df_filtered[date_col] <= t2_ts)
            ]

    return df_filtered


def calculate_inventory_summary(inv_df):
    """计算库存汇总数据"""
    if inv_df is None or inv_df.empty:
        return {
            "Total Inventory Value": 0,
            "Total Retail Value": 0,
            "Profit": 0,
            "Profit Margin": "0.0%"
        }

    df = inv_df.copy()

    # 1. 过滤掉负数、0、空值的库存和成本
    df["Quantity"] = pd.to_numeric(df["Current Quantity Vie Market & Bar"], errors="coerce")
    df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitCost"] > 0)].copy()

    if df.empty:
        return {
            "Total Inventory Value": 0,
            "Total Retail Value": 0,
            "Profit": 0,
            "Profit Margin": "0.0%"
        }

    # 2. 处理单位成本
    df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce").fillna(0)

    # 3. 计算 Inventory Value
    df["Inventory Value"] = df["UnitCost"] * df["Quantity"]
    total_inventory_value = df["Inventory Value"].sum()

    # 4. 计算 Total Retail Value
    def calc_single_retail(row):
        try:
            O, AA, tax = row["Price"], row["Quantity"], str(row["Tax - GST (10%)"]).strip().upper()
            return (O / 11 * 10) * AA if tax == "Y" else O * AA
        except KeyError:
            return row["Price"] * row["Quantity"]

    df["Single Retail Value"] = df.apply(calc_single_retail, axis=1)
    total_retail_value = df["Single Retail Value"].sum()

    # 5. 计算 Profit 和 Profit Margin
    profit = total_retail_value - total_inventory_value
    profit_margin = (profit / total_retail_value * 100) if total_retail_value > 0 else 0

    # 四舍五入
    total_inventory_value = round(total_inventory_value)
    total_retail_value = round(total_retail_value)
    profit = round(profit)
    total_inventory_value = int(total_inventory_value)

    return {
        "Total Inventory Value": total_inventory_value,
        "Total Retail Value": total_retail_value,
        "Profit": profit,
        "Profit Margin": f"{profit_margin:.1f}%"
    }


def show_inventory(tx, inventory: pd.DataFrame):
    # === 全局样式：参考 high_level 的样式设置 ===
    st.markdown("""
    <style>
    /* 去掉标题之间的空白 */
    div.block-container h1, 
    div.block-container h2, 
    div.block-container h3, 
    div.block-container h4,
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

    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    /* 让表格文字左对齐 */
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }

    /* 让 Current Quantity 输入框和多选框对齐 */
    div[data-testid*="stNumberInput"] {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    div[data-testid*="stNumberInput"] label {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    /* 统一多选框和输入框的垂直对齐 */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        align-items: start !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === 标题样式参考 high_level ===
    st.markdown("<h2 style='font-size:24px; font-weight:700;'>📦 Product Mix & Inventory Optimization</h2>",
                unsafe_allow_html=True)

    if tx.empty:
        st.info("No transaction data available")
        return

    if inventory is None or inventory.empty:
        st.info("No inventory data available")
        return

    inv = inventory.copy()

    # ---- 💰 Inventory Valuation Analysis ----
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>💰 Inventory Valuation Analysis</h3>",
                unsafe_allow_html=True)

    # === 修改：使用与 sales_report.py 相同的三列布局 ===
    col_date, col_search, col_select, _ = st.columns([1, 1, 1.8, 3.5])

    with col_date:
        # 获取可用的日期（从库存数据中提取）
        if "source_date" in inv.columns:
            available_dates = sorted(pd.to_datetime(inv["source_date"]).dt.date.unique(), reverse=True)
        elif "date" in inv.columns:
            available_dates = sorted(pd.to_datetime(inv["date"]).dt.date.unique(), reverse=True)
        else:
            available_dates = []

        # 将日期格式改为欧洲格式显示
        available_dates_formatted = [date.strftime('%d/%m/%Y') for date in available_dates]

        # === 修复：使用正确的 CSS 选择器设置日期选择框宽度 ===
        st.markdown("""
        <style>
        /* 仅影响日期选择框：通过label名称或key限定 */
        div[data-testid*="stSelectbox"][aria-label="Choose date"],
        div[data-testid*="stSelectbox"][data-baseweb="select"][aria-label="Choose date"] {
            width: 18ch !important;
            min-width: 18ch !important;
            max-width: 18ch !important;
        }
        </style>
        """, unsafe_allow_html=True)

        selected_date_formatted = st.selectbox("Choose date", available_dates_formatted)

        # 将选择的日期转换回日期对象
        selected_date = pd.to_datetime(selected_date_formatted, format='%d/%m/%Y').date()

    with col_search:
        # 搜索关键词输入框
        st.markdown("""
        <style>
        div[data-testid*="cat_search_term"] {
            width: 25ch !important;
            min-width: 25ch !important;
        }
        div[data-testid*="cat_search_term"] input {
            width: 25ch !important;
            min-width: 25ch !important;
        }
        </style>
        """, unsafe_allow_html=True)

        cat_search_term = st.text_input(
            "🔍 Input",
            placeholder="",
            key="cat_search_term"
        )

    with col_select:
        all_items = sorted(inv["Item Name"].fillna("Unknown").unique().tolist()) if "Item Name" in inv.columns else []
        bar_cats = ["Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"]

        # 根据搜索词过滤选项
        if cat_search_term:
            search_lower = cat_search_term.lower()
            filtered_options = [item for item in (all_items + ["bar", "retail"]) if
                                search_lower in str(item).lower()]
            item_count_text = f"{len(filtered_options)} categories"
        else:
            filtered_options = all_items + ["bar", "retail"]
            item_count_text = f"{len(filtered_options)} items"

        # === 修改：设置多选框宽度与输入框对齐 ===
        categories = persisting_multiselect(
            f"Select Items ({item_count_text})",
            filtered_options,
            key="inv_cats_box",
            width_chars=25  # 设置为与输入框相同的宽度
        )

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    # 移除原有的时间范围选择逻辑，现在使用单一日期
    time_range = []  # 清空时间范围，因为现在只用单一日期
    custom_dates_selected = False
    t1 = None
    t2 = None

    # ---- 📊 Selected Categories Table ----
    if categories:
        st.markdown("<h3 style='font-size:20px; font-weight:700;'>📊 Selected Categories Inventory</h3>",
                    unsafe_allow_html=True)

        # 获取选定日期的库存数据
        if "source_date" in inv.columns or "date" in inv.columns:
            date_col = "source_date" if "source_date" in inv.columns else "date"
            inv_with_date = inv.copy()
            inv_with_date[date_col] = pd.to_datetime(inv_with_date[date_col], errors="coerce")
            # 筛选选定日期的数据
            filtered_inv = inv_with_date[inv_with_date[date_col].dt.date == selected_date]
        else:
            filtered_inv = inv.copy()

        # 根据选择的分类筛选数据
        if "bar" in categories:
            # 如果选择了bar，显示所有bar分类的商品
            bar_items = filtered_inv[filtered_inv["Item Name"].isin(bar_cats)]
            cat_filtered_inv = bar_items
        elif "retail" in categories:
            # 如果选择了retail，显示非bar分类的商品
            retail_items = filtered_inv[~filtered_inv["Item Name"].isin(bar_cats)]
            cat_filtered_inv = retail_items
        else:
            # 显示具体选择的分类
            cat_filtered_inv = filtered_inv[filtered_inv["Item Name"].isin(categories)]

        if not cat_filtered_inv.empty:
            # 准备显示数据 - 使用与Low Stock Alerts相同的列格式
            display_df = cat_filtered_inv.copy()

            # 确保数值列是数字类型
            display_df["Current Quantity Vie Market & Bar"] = pd.to_numeric(
                display_df["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0)
            display_df["Price"] = pd.to_numeric(display_df["Price"], errors="coerce").fillna(0)
            display_df["Default Unit Cost"] = pd.to_numeric(display_df["Default Unit Cost"], errors="coerce").fillna(0)

            # 计算 Total Inventory (使用绝对值)
            display_df["Total Inventory"] = display_df["Default Unit Cost"] * abs(
                display_df["Current Quantity Vie Market & Bar"])

            # 计算 Total Retail
            def calc_retail(row):
                O, AA, tax = row["Price"], abs(row["Current Quantity Vie Market & Bar"]), str(
                    row["Tax - GST (10%)"]).strip().upper()
                return (O / 11 * 10) * AA if tax == "Y" else O * AA

            display_df["Total Retail"] = display_df.apply(calc_retail, axis=1)

            # 计算 Profit
            display_df["Profit"] = display_df["Total Retail"] - display_df["Total Inventory"]

            # 所有数值列先四舍五入处理浮点数精度问题
            display_df["Total Inventory"] = display_df["Total Inventory"].round(2)
            display_df["Total Retail"] = display_df["Total Retail"].round(2)
            display_df["Profit"] = display_df["Profit"].round(2)

            # 计算 Profit Margin
            display_df["Profit Margin"] = (display_df["Profit"] / display_df["Total Retail"] * 100).fillna(0)
            display_df["Profit Margin"] = display_df["Profit Margin"].map(lambda x: f"{x:.1f}%")

            # 计算过去4周的Net Sales
            selected_date_ts = pd.Timestamp(selected_date)

            # === 新逻辑：按 Item Name 连接 transaction 表 ===
            tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
            past_4w_start = selected_date_ts - pd.Timedelta(days=28)
            recent_tx = tx[(tx["Datetime"] >= past_4w_start) & (tx["Datetime"] <= selected_date_ts)].copy()

            recent_tx["Item"] = recent_tx["Item"].astype(str).str.strip()
            recent_tx["Net Sales"] = pd.to_numeric(recent_tx["Net Sales"], errors="coerce").fillna(0)

            item_sales_4w = (
                recent_tx.groupby("Item")["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Item": "Item Name", "Net Sales": "Net Sale 4W"})
            )

            display_df = display_df.merge(item_sales_4w, on="Item Name", how="left")
            display_df["Velocity"] = display_df.apply(
                lambda r: round(r["Total Retail"] / r["Net Sale 4W"], 2)
                if pd.notna(r["Net Sale 4W"]) and r["Net Sale 4W"] > 0
                else "-",
                axis=1
            )

            vel_numeric = pd.to_numeric(display_df["Velocity"], errors="coerce")
            display_df["Velocity"] = vel_numeric.round(1).where(vel_numeric.notna(), display_df["Velocity"])

            # 重命名 Current Quantity Vie Market & Bar 列为 Current Quantity
            display_df = display_df.rename(columns={"Current Quantity Vie Market & Bar": "Current Quantity"})

            # === 修改：所有 Current Quantity 展示绝对值 ===
            display_df["Current Quantity"] = display_df["Current Quantity"].abs()

            # 选择要显示的列
            display_columns = []
            if "Item Name" in display_df.columns:
                display_columns.append("Item Name")
            if "Item Variation Name" in display_df.columns:
                display_columns.append("Item Variation Name")
            if "SKU" in display_df.columns:
                display_columns.append("SKU")

            display_columns.extend(
                ["Current Quantity", "Total Inventory", "Total Retail", "Profit", "Profit Margin", "Velocity"])

            # 确保 SKU 列完整显示（不使用科学记数法）
            if "SKU" in display_df.columns:
                display_df["SKU"] = display_df["SKU"].astype(str)

            # 特殊处理：Velocity 为0、无限大、空值或无效值用 '-' 替换
            def clean_velocity(x):
                if pd.isna(x) or x == 0 or x == float('inf') or x == float('-inf'):
                    return '-'
                return x

            display_df["Velocity"] = display_df["Velocity"].apply(clean_velocity)

            # Total Retail, Total Inventory, Profit 列为0的值用 '-' 替换
            display_df["Total Retail"] = display_df["Total Retail"].apply(lambda x: '-' if x == 0 else x)
            display_df["Total Inventory"] = display_df["Total Inventory"].apply(lambda x: '-' if x == 0 else x)
            display_df["Profit"] = display_df["Profit"].apply(lambda x: '-' if x == 0 else x)

            # 其他空值用字符 '-' 替换
            for col in display_columns:
                if col in display_df.columns:
                    if col not in ["Total Retail", "Total Inventory", "Profit", "Velocity"]:  # 这些列已经特殊处理过
                        display_df[col] = display_df[col].fillna('-')

            column_config = {
                'Item Name': st.column_config.Column(width=150),
                'Item Variation Name': st.column_config.Column(width=50),

                'SKU': st.column_config.Column(width=100),
                'Current Quantity': st.column_config.Column(width=110),
                'Total Inventory': st.column_config.Column(width=100),
                'Total Retail': st.column_config.Column(width=80),
                'Profit': st.column_config.Column(width=50),
                'Profit Margin': st.column_config.Column(width=90),
                'Velocity': st.column_config.Column(width=60),
            }

            st.dataframe(
                display_df[display_columns],
                column_config=column_config,
                use_container_width=False
            )
        else:
            st.info("No items found for selected categories on the chosen date.")

    # ---- 📊 Inventory Summary Table - 参考 Summary Table 格式 ----
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>📊 Inventory Summary</h3>", unsafe_allow_html=True)

    # 获取选定日期的库存数据
    if "source_date" in inv.columns or "date" in inv.columns:
        date_col = "source_date" if "source_date" in inv.columns else "date"
        inv_with_date = inv.copy()
        inv_with_date[date_col] = pd.to_datetime(inv_with_date[date_col], errors="coerce")
        # 筛选选定日期的数据
        filtered_inv = inv_with_date[inv_with_date[date_col].dt.date == selected_date]
        summary_data = calculate_inventory_summary(filtered_inv)
    else:
        summary_data = calculate_inventory_summary(inv)

    # 显示选定日期 - 参考 high_level 的格式
    st.markdown(
        f"<h4 style='font-size:16px; font-weight:700;'>Selected Date: {selected_date.strftime('%d/%m/%Y')}</h4>",
        unsafe_allow_html=True)

    # 创建类似 Summary Table 格式的数据框
    summary_table_data = {
        'Metric': ['Total Inventory Value', 'Total Retail Value', 'Profit', 'Profit Margin'],
        'Value': [
            f"${summary_data['Total Inventory Value']:,}",
            f"${summary_data['Total Retail Value']:,}",
            f"${summary_data['Profit']:,}",
            summary_data['Profit Margin']
        ]
    }

    df_summary = pd.DataFrame(summary_table_data)

    # 设置列配置 - 参考 sales_report 格式，不强制占满一行
    column_config = {
        'Metric': st.column_config.Column(width=135),
        'Value': st.column_config.Column(width=70),
    }

    # 显示表格
    st.dataframe(
        df_summary,
        column_config=column_config,
        hide_index=True,
        use_container_width=False
    )

    st.markdown("---")

    # ---- 1) Inventory Diagnosis: Restock / Clearance ----
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>1) Inventory Diagnosis: Restock / Clearance Needed</h3>",
                unsafe_allow_html=True)
    qty_col = detect_store_current_qty_col(inv)

    item_col = "Item Name" if "Item Name" in inv.columns else "Item"
    variation_col = "Item Variation Name" if "Item Variation Name" in inv.columns else None
    sku_col = "SKU" if "SKU" in inv.columns else None

    if variation_col:
        inv["display_name"] = inv[item_col].astype(str) + " - " + inv[variation_col].astype(str)
    else:
        inv["display_name"] = inv[item_col].astype(str)

    if sku_col:
        inv["option_key"] = inv["display_name"] + " (SKU:" + inv[sku_col].astype(str) + ")"
    else:
        inv["option_key"] = inv["display_name"]

    # === 生成补货表 ===
    need_restock = filtered_inv.copy()

    # ✅ 确保存在 option_key 列
    if "option_key" not in need_restock.columns:
        if "Item Name" in need_restock.columns:
            item_col = "Item Name"
        else:
            item_col = "Item"
        variation_col = "Item Variation Name" if "Item Variation Name" in need_restock.columns else None
        sku_col = "SKU" if "SKU" in need_restock.columns else None

        if variation_col:
            need_restock["display_name"] = need_restock[item_col].astype(str) + " - " + need_restock[
                variation_col].astype(str)
        else:
            need_restock["display_name"] = need_restock[item_col].astype(str)

        if sku_col:
            need_restock["option_key"] = need_restock["display_name"] + " (SKU:" + need_restock[sku_col].astype(
                str) + ")"
        else:
            need_restock["option_key"] = need_restock["display_name"]

    # ✅ 再按库存量筛选需要补货的
    need_restock = need_restock[pd.to_numeric(need_restock[qty_col], errors="coerce").fillna(0) < 0].copy()

    if not need_restock.empty:
        options = sorted(need_restock["option_key"].unique())

        # === 修改：添加空白行确保水平对齐 ===
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        # === 修改：参考 sales_report 的布局，使用三列布局 ===
        col_select_restock, col_threshold_restock, _ = st.columns([1.8, 1, 4.2])

        with col_select_restock:
            selected_items = persisting_multiselect(
                f"Select Items ({len(options)} items)",
                options,
                key="restock_filter",
                default=[],
                width_chars=25  # 与输入框对齐
            )

        with col_threshold_restock:
            # === 修改：添加空白标签确保垂直对齐 ===
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            # === 修改：改为单选框，直接输入数字作为阈值 ===
            max_qty = int(need_restock[qty_col].abs().max())
            threshold_value = st.number_input(
                "Current Quantity ≤",
                min_value=0,
                max_value=max_qty,
                value=max_qty,
                key="restock_threshold",
                help="Enter threshold value"
            )

        df_show = need_restock.copy()
        # ✅ 使用缺货数量的绝对值
        df_show["restock_needed"] = pd.to_numeric(df_show[qty_col], errors="coerce").fillna(0).abs()

        # 应用阈值筛选 - 筛选小于等于输入值的项目
        df_show = df_show[df_show["restock_needed"] <= threshold_value]

        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_show = df_show[df_show["SKU"].astype(str).isin(selected_skus)]
            else:
                df_show = df_show[df_show["display_name"].isin(selected_items)]

        if not df_show.empty:
            # === 修改：准备显示数据，参考 Low Stock Alerts 格式 ===
            display_restock = df_show.copy()

            # 确保数值列是数字类型
            display_restock["Current Quantity Vie Market & Bar"] = pd.to_numeric(
                display_restock["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0)
            display_restock["Price"] = pd.to_numeric(display_restock["Price"], errors="coerce").fillna(0)
            display_restock["Default Unit Cost"] = pd.to_numeric(display_restock["Default Unit Cost"],
                                                                 errors="coerce").fillna(0)

            # 计算 Total Inventory (使用绝对值)
            display_restock["Total Inventory"] = display_restock["Default Unit Cost"] * abs(
                display_restock["Current Quantity Vie Market & Bar"])

            # 计算 Total Retail
            def calc_retail(row):
                O, AA, tax = row["Price"], abs(row["Current Quantity Vie Market & Bar"]), str(
                    row["Tax - GST (10%)"]).strip().upper()
                return (O / 11 * 10) * AA if tax == "Y" else O * AA

            display_restock["Total Retail"] = display_restock.apply(calc_retail, axis=1)

            # 计算 Profit
            display_restock["Profit"] = display_restock["Total Retail"] - display_restock["Total Inventory"]

            # 所有数值列先四舍五入处理浮点数精度问题
            display_restock["Total Inventory"] = display_restock["Total Inventory"].round(2)
            display_restock["Total Retail"] = display_restock["Total Retail"].round(2)
            display_restock["Profit"] = display_restock["Profit"].round(2)

            # 计算 Profit Margin
            display_restock["Profit Margin"] = (
                    display_restock["Profit"] / display_restock["Total Retail"] * 100).fillna(0)
            display_restock["Profit Margin"] = display_restock["Profit Margin"].map(lambda x: f"{x:.1f}%")

            # 计算过去4周的Net Sales
            selected_date_ts = pd.Timestamp(selected_date)

            tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
            past_4w_start = selected_date_ts - pd.Timedelta(days=28)
            recent_tx = tx[(tx["Datetime"] >= past_4w_start) & (tx["Datetime"] <= selected_date_ts)].copy()

            recent_tx["Item"] = recent_tx["Item"].astype(str).str.strip()
            recent_tx["Net Sales"] = pd.to_numeric(recent_tx["Net Sales"], errors="coerce").fillna(0)

            item_sales_4w = (
                recent_tx.groupby("Item")["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Item": "Item Name", "Net Sales": "Net Sale 4W"})
            )

            display_restock = display_restock.merge(item_sales_4w, on="Item Name", how="left")
            display_restock["Velocity"] = display_restock.apply(
                lambda r: round(r["Total Retail"] / r["Net Sale 4W"], 2)
                if pd.notna(r["Net Sale 4W"]) and r["Net Sale 4W"] > 0
                else "-",
                axis=1
            )

            # Velocity 四舍五入保留一位小数
            vel_numeric = pd.to_numeric(display_restock["Velocity"], errors="coerce")
            display_restock["Velocity"] = vel_numeric.round(1).where(vel_numeric.notna(), display_restock["Velocity"])

            # 重命名 Current Quantity Vie Market & Bar 列为 Current Quantity
            display_restock = display_restock.rename(columns={"Current Quantity Vie Market & Bar": "Current Quantity"})

            # === 修改：所有 Current Quantity 展示绝对值 ===
            display_restock["Current Quantity"] = display_restock["Current Quantity"].abs()

            # 选择要显示的列
            display_columns = []
            if "Item Name" in display_restock.columns:
                display_columns.append("Item Name")
            if "Item Variation Name" in display_restock.columns:
                display_columns.append("Item Variation Name")
            if "SKU" in display_restock.columns:
                display_columns.append("SKU")

            display_columns.extend(
                ["Current Quantity", "Total Inventory", "Total Retail", "Profit", "Profit Margin", "Velocity"])

            # 确保 SKU 列完整显示（不使用科学记数法）
            if "SKU" in display_restock.columns:
                display_restock["SKU"] = display_restock["SKU"].astype(str)

            # 特殊处理：Velocity 为0、无限大、空值或无效值用 '-' 替换
            def clean_velocity(x):
                if pd.isna(x) or x == 0 or x == float('inf') or x == float('-inf'):
                    return '-'
                return x

            display_restock["Velocity"] = display_restock["Velocity"].apply(clean_velocity)

            # Total Retail, Total Inventory, Profit 列为0的值用 '-' 替换
            display_restock["Total Retail"] = display_restock["Total Retail"].apply(lambda x: '-' if x == 0 else x)
            display_restock["Total Inventory"] = display_restock["Total Inventory"].apply(
                lambda x: '-' if x == 0 else x)
            display_restock["Profit"] = display_restock["Profit"].apply(lambda x: '-' if x == 0 else x)

            # 其他空值用字符 '-' 替换
            for col in display_columns:
                if col in display_restock.columns:
                    if col not in ["Total Retail", "Total Inventory", "Profit", "Velocity"]:  # 这些列已经特殊处理过
                        display_restock[col] = display_restock[col].fillna('-')

            # === 修改：设置列宽配置，参考 sales_report 格式 ===
            column_config = {
                'Item Name': st.column_config.Column(width=150),
                'Item Variation Name': st.column_config.Column(width=50),

                'SKU': st.column_config.Column(width=100),
                'Current Quantity': st.column_config.Column(width=110),
                'Total Inventory': st.column_config.Column(width=100),
                'Total Retail': st.column_config.Column(width=80),
                'Profit': st.column_config.Column(width=50),
                'Profit Margin': st.column_config.Column(width=90),
                'Velocity': st.column_config.Column(width=60),
            }

            st.dataframe(
                display_restock[display_columns],
                column_config=column_config,
                use_container_width=False
            )
        else:
            st.info("No matching items to restock.")
    else:
        st.success("No items need restocking.")

    clear_threshold = 50
    # === 生成需要清仓表（high stock items） ===
    need_clear = filtered_inv.copy()

    # ✅ 确保存在 option_key 列
    if "option_key" not in need_clear.columns:
        if "Item Name" in need_clear.columns:
            item_col = "Item Name"
        else:
            item_col = "Item"
        variation_col = "Item Variation Name" if "Item Variation Name" in need_clear.columns else None
        sku_col = "SKU" if "SKU" in need_clear.columns else None

        if variation_col:
            need_clear["display_name"] = need_clear[item_col].astype(str) + " - " + need_clear[variation_col].astype(
                str)
        else:
            need_clear["display_name"] = need_clear[item_col].astype(str)

        if sku_col:
            need_clear["option_key"] = need_clear["display_name"] + " (SKU:" + need_clear[sku_col].astype(str) + ")"
        else:
            need_clear["option_key"] = need_clear["display_name"]

    # ✅ 过滤大库存行（例如超过 clear_threshold）
    need_clear = need_clear[pd.to_numeric(need_clear[qty_col], errors="coerce").fillna(0) >= clear_threshold].copy()

    if not need_clear.empty:
        options = sorted(need_clear["option_key"].unique())

        # === 修改：添加空白行确保水平对齐 ===
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        # === 修改：参考 sales_report 的布局，使用三列布局 ===
        col_select_clear, col_threshold_clear, _ = st.columns([1.8, 1, 4.2])

        with col_select_clear:
            selected_items = persisting_multiselect(
                f"Select Items ({len(options)} items)",
                options,
                key="clear_filter",
                default=[],
                width_chars=25  # 与输入框对齐
            )

        with col_threshold_clear:
            # === 修改：添加空白标签确保垂直对齐 ===
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            # === 修改：改为单选框，直接输入数字作为阈值 ===
            max_qty = int(need_clear[qty_col].max())
            threshold_value = st.number_input(
                "Current Quantity ≥",
                min_value=clear_threshold,
                max_value=max_qty,
                value=clear_threshold,
                key="clear_threshold",
                help="Enter threshold value"
            )

        df_clear = need_clear.copy()
        df_clear["current_qty"] = pd.to_numeric(df_clear[qty_col], errors="coerce").fillna(0)

        # 应用阈值筛选 - 筛选大于等于输入值的项目
        df_clear = df_clear[df_clear["current_qty"] >= threshold_value]

        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_clear = df_clear[df_clear["SKU"].astype(str).isin(selected_skus)]
            else:
                df_clear = df_clear[df_clear["display_name"].isin(selected_items)]

        if not df_clear.empty:
            # === 修改：只展示top10的items，图表宽度为一半 ===
            top_10_clear = df_clear.nlargest(10, "current_qty")
            fig_clear = px.bar(top_10_clear, x="display_name", y="current_qty",
                               title="Items Needing Clearance (units) - Top 10",
                               labels={"current_qty": "Stock Quantity (units)", "display_name": "Item Name"})
            fig_clear.update_layout(width=600)  # 设置图表宽度为现在的一半
            st.plotly_chart(fig_clear, use_container_width=False)

            # 计算所需的列
            df_clear_display = df_clear.copy()

            # 确保数值列是数字类型
            df_clear_display["Current Quantity Vie Market & Bar"] = pd.to_numeric(
                df_clear_display["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0)
            df_clear_display["Price"] = pd.to_numeric(df_clear_display["Price"], errors="coerce").fillna(0)
            df_clear_display["Default Unit Cost"] = pd.to_numeric(df_clear_display["Default Unit Cost"],
                                                                  errors="coerce").fillna(0)

            # 计算 Total Inventory (使用绝对值)
            df_clear_display["Total Inventory"] = df_clear_display["Default Unit Cost"] * abs(
                df_clear_display["Current Quantity Vie Market & Bar"])

            # 计算 Total Retail
            def calc_retail(row):
                O, AA, tax = row["Price"], abs(row["Current Quantity Vie Market & Bar"]), str(
                    row["Tax - GST (10%)"]).strip().upper()
                return (O / 11 * 10) * AA if tax == "Y" else O * AA

            df_clear_display["Total Retail"] = df_clear_display.apply(calc_retail, axis=1)

            # 计算 Profit
            df_clear_display["Profit"] = df_clear_display["Total Retail"] - df_clear_display["Total Inventory"]

            # 所有数值列先四舍五入处理浮点数精度问题
            df_clear_display["Total Inventory"] = df_clear_display["Total Inventory"].round(2)
            df_clear_display["Total Retail"] = df_clear_display["Total Retail"].round(2)
            df_clear_display["Profit"] = df_clear_display["Profit"].round(2)

            # 计算 Profit Margin
            df_clear_display["Profit Margin"] = (
                    df_clear_display["Profit"] / df_clear_display["Total Retail"] * 100).fillna(0)
            df_clear_display["Profit Margin"] = df_clear_display["Profit Margin"].map(lambda x: f"{x:.1f}%")

            # 计算过去4周的Net Sales
            selected_date_ts = pd.Timestamp(selected_date)

            tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
            past_4w_start = selected_date_ts - pd.Timedelta(days=28)
            recent_tx = tx[(tx["Datetime"] >= past_4w_start) & (tx["Datetime"] <= selected_date_ts)].copy()

            recent_tx["Item"] = recent_tx["Item"].astype(str).str.strip()
            recent_tx["Net Sales"] = pd.to_numeric(recent_tx["Net Sales"], errors="coerce").fillna(0)

            item_sales_4w = (
                recent_tx.groupby("Item")["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Item": "Item Name", "Net Sales": "Net Sale 4W"})
            )

            df_clear_display = df_clear_display.merge(item_sales_4w, on="Item Name", how="left")
            df_clear_display["Velocity"] = df_clear_display.apply(
                lambda r: round(r["Total Retail"] / r["Net Sale 4W"], 2)
                if pd.notna(r["Net Sale 4W"]) and r["Net Sale 4W"] > 0
                else "-",
                axis=1
            )

            vel_numeric = pd.to_numeric(df_clear_display["Velocity"], errors="coerce")
            df_clear_display["Velocity"] = vel_numeric.round(1).where(vel_numeric.notna(), df_clear_display["Velocity"])

            # 重命名 Current Quantity Vie Market & Bar 列为 Current Quantity
            df_clear_display = df_clear_display.rename(
                columns={"Current Quantity Vie Market & Bar": "Current Quantity"})

            # === 修改：所有 Current Quantity 展示绝对值 ===
            df_clear_display["Current Quantity"] = df_clear_display["Current Quantity"].abs()

            # 选择要显示的列
            display_columns = []
            if "Item Name" in df_clear_display.columns:
                display_columns.append("Item Name")
            if "Item Variation Name" in df_clear_display.columns:
                display_columns.append("Item Variation Name")
            if "SKU" in df_clear_display.columns:
                display_columns.append("SKU")

            display_columns.extend(
                ["Current Quantity", "Total Inventory", "Total Retail", "Profit", "Profit Margin", "Velocity"])

            # 确保 SKU 列完整显示（不使用科学记数法）
            if "SKU" in df_clear_display.columns:
                df_clear_display["SKU"] = df_clear_display["SKU"].astype(str)

            # 特殊处理：Velocity 为0、无限大、空值或无效值用 '-' 替换
            def clean_velocity(x):
                if pd.isna(x) or x == 0 or x == float('inf') or x == float('-inf'):
                    return '-'
                return x

            df_clear_display["Velocity"] = df_clear_display["Velocity"].apply(clean_velocity)

            # Total Retail, Total Inventory, Profit 列为0的值用 '-' 替换
            df_clear_display["Total Retail"] = df_clear_display["Total Retail"].apply(lambda x: '-' if x == 0 else x)
            df_clear_display["Total Inventory"] = df_clear_display["Total Inventory"].apply(
                lambda x: '-' if x == 0 else x)
            df_clear_display["Profit"] = df_clear_display["Profit"].apply(lambda x: '-' if x == 0 else x)

            # 其他空值用字符 '-' 替换
            for col in display_columns:
                if col in df_clear_display.columns:
                    if col not in ["Total Retail", "Total Inventory", "Profit", "Velocity"]:  # 这些列已经特殊处理过
                        df_clear_display[col] = df_clear_display[col].fillna('-')

            column_config = {
                'Item Name': st.column_config.Column(width=150),
                'Item Variation Name': st.column_config.Column(width=50),

                'SKU': st.column_config.Column(width=100),
                'Current Quantity': st.column_config.Column(width=110),
                'Total Inventory': st.column_config.Column(width=100),
                'Total Retail': st.column_config.Column(width=80),
                'Profit': st.column_config.Column(width=50),
                'Profit Margin': st.column_config.Column(width=90),
                'Velocity': st.column_config.Column(width=60),
            }

            st.dataframe(
                df_clear_display[display_columns],
                column_config=column_config,
                use_container_width=False
            )
        else:
            st.info("No matching items needing clearance.")
    else:
        st.success("No items need clearance.")

    st.markdown("---")

    # ---- 2) Low Stock Alerts ----
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>2) Low Stock Alerts</h3>", unsafe_allow_html=True)

    # === 生成低库存表 ===
    low_stock = filtered_inv.copy()

    # ✅ 确保存在 option_key 列
    if "option_key" not in low_stock.columns:
        if "Item Name" in low_stock.columns:
            item_col = "Item Name"
        else:
            item_col = "Item"
        variation_col = "Item Variation Name" if "Item Variation Name" in low_stock.columns else None
        sku_col = "SKU" if "SKU" in low_stock.columns else None

        if variation_col:
            low_stock["display_name"] = low_stock[item_col].astype(str) + " - " + low_stock[variation_col].astype(str)
        else:
            low_stock["display_name"] = low_stock[item_col].astype(str)

        if sku_col:
            low_stock["option_key"] = low_stock["display_name"] + " (SKU:" + low_stock[sku_col].astype(str) + ")"
        else:
            low_stock["option_key"] = low_stock["display_name"]

    # ✅ 过滤 1–20 单位的低库存行
    low_stock = low_stock[pd.to_numeric(low_stock[qty_col], errors="coerce").fillna(0).between(1, 20)].copy()

    if not low_stock.empty:
        options = sorted(low_stock["option_key"].unique())

        # === 修改：参考 Inventory Valuation Analysis 的布局，使用四列布局 ===
        col_search_low, col_select_low, col_threshold_low, _ = st.columns([1, 1.8, 1, 3.2])

        with col_search_low:
            st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)
            # === 修改：添加二级搜索框 ===
            low_stock_search_term = st.text_input(
                "🔍 Search",
                placeholder="Search items...",
                key="low_stock_search_term"
            )

        with col_select_low:
            # 根据搜索词过滤选项
            if low_stock_search_term:
                search_lower = low_stock_search_term.lower()
                filtered_options = [item for item in options if search_lower in str(item).lower()]
                item_count_text = f"{len(filtered_options)} items"
            else:
                filtered_options = options
                item_count_text = f"{len(options)} items"

            selected_items = persisting_multiselect(
                f"Select Items ({item_count_text})",
                filtered_options,
                key="low_stock_filter",
                default=[],
                width_chars=25  # 与输入框对齐
            )

        with col_threshold_low:
            #current quantity
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            # === 修改：改为单选框，直接输入数字作为阈值 ===
            max_qty = int(low_stock[qty_col].max())
            threshold_value = st.number_input(
                "Current Quantity ≤",
                min_value=1,
                max_value=20,
                value=20,
                key="low_stock_threshold",
                help="Enter threshold value"
            )

        df_low = low_stock.copy()
        df_low["current_qty"] = pd.to_numeric(df_low[qty_col], errors="coerce").fillna(0)

        # 应用阈值筛选 - 筛选小于等于输入值的项目
        df_low = df_low[df_low["current_qty"] <= threshold_value]

        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_low = df_low[df_low["SKU"].astype(str).isin(selected_skus)]
            else:
                df_low = df_low[df_low["display_name"].isin(selected_items)]

        if not df_low.empty:
            # === 修改：添加top10的柱形图 ===
            top_10_low = df_low.nlargest(10, "current_qty")
            fig_low = px.bar(top_10_low, x="display_name", y="current_qty",
                             title="Low Stock Items (units) - Top 10",
                             labels={"current_qty": "Stock Quantity (units)", "display_name": "Item Name"})
            fig_low.update_layout(width=600)  # 设置图表宽度
            st.plotly_chart(fig_low, use_container_width=False)

            # 计算所需的列
            df_low_display = df_low.copy()

            # 确保数值列是数字类型
            df_low_display["Current Quantity Vie Market & Bar"] = pd.to_numeric(
                df_low_display["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0)
            df_low_display["Price"] = pd.to_numeric(df_low_display["Price"], errors="coerce").fillna(0)
            df_low_display["Default Unit Cost"] = pd.to_numeric(df_low_display["Default Unit Cost"],
                                                                errors="coerce").fillna(0)

            # 计算 Total Inventory (使用绝对值)
            df_low_display["Total Inventory"] = df_low_display["Default Unit Cost"] * abs(
                df_low_display["Current Quantity Vie Market & Bar"])

            # 计算 Total Retail
            def calc_retail(row):
                O, AA, tax = row["Price"], abs(row["Current Quantity Vie Market & Bar"]), str(
                    row["Tax - GST (10%)"]).strip().upper()
                return (O / 11 * 10) * AA if tax == "Y" else O * AA

            df_low_display["Total Retail"] = df_low_display.apply(calc_retail, axis=1)

            # 计算 Profit
            df_low_display["Profit"] = df_low_display["Total Retail"] - df_low_display["Total Inventory"]

            # 所有数值列先四舍五入处理浮点数精度问题
            df_low_display["Total Inventory"] = df_low_display["Total Inventory"].round(2)
            df_low_display["Total Retail"] = df_low_display["Total Retail"].round(2)
            df_low_display["Profit"] = df_low_display["Profit"].round(2)

            # 计算 Profit Margin
            df_low_display["Profit Margin"] = (df_low_display["Profit"] / df_low_display["Total Retail"] * 100).fillna(
                0)
            df_low_display["Profit Margin"] = df_low_display["Profit Margin"].map(lambda x: f"{x:.1f}%")

            # 计算过去4周的Net Sales
            selected_date_ts = pd.Timestamp(selected_date)

            # === 新 Velocity 逻辑：按 Item Name 连接 transaction 表 ===
            tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
            past_4w_start = selected_date_ts - pd.Timedelta(days=28)
            recent_tx = tx[(tx["Datetime"] >= past_4w_start) & (tx["Datetime"] <= selected_date_ts)].copy()

            recent_tx["Item"] = recent_tx["Item"].astype(str).str.strip()
            recent_tx["Net Sales"] = pd.to_numeric(recent_tx["Net Sales"], errors="coerce").fillna(0)

            item_sales_4w = (
                recent_tx.groupby("Item")["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Item": "Item Name", "Net Sales": "Net Sale 4W"})
            )

            df_low_display = df_low_display.merge(item_sales_4w, on="Item Name", how="left")
            df_low_display["Velocity"] = df_low_display.apply(
                lambda r: round(r["Total Retail"] / r["Net Sale 4W"], 2)
                if pd.notna(r["Net Sale 4W"]) and r["Net Sale 4W"] > 0
                else "-",
                axis=1
            )

            # Velocity 四舍五入保留一位小数
            vel_numeric = pd.to_numeric(df_low_display["Velocity"], errors="coerce")
            df_low_display["Velocity"] = vel_numeric.round(1).where(vel_numeric.notna(), df_low_display["Velocity"])

            # 重命名 Current Quantity Vie Market & Bar 列为 Current Quantity
            df_low_display = df_low_display.rename(columns={"Current Quantity Vie Market & Bar": "Current Quantity"})

            # === 修改：所有 Current Quantity 展示绝对值 ===
            df_low_display["Current Quantity"] = df_low_display["Current Quantity"].abs()

            # 选择要显示的列
            display_columns = []
            if "Item Name" in df_low_display.columns:
                display_columns.append("Item Name")
            if "Item Variation Name" in df_low_display.columns:
                display_columns.append("Item Variation Name")
            if "SKU" in df_low_display.columns:
                display_columns.append("SKU")

            display_columns.extend(
                ["Current Quantity", "Total Inventory", "Total Retail", "Profit", "Profit Margin", "Velocity"])

            # 确保 SKU 列完整显示（不使用科学记数法）
            if "SKU" in df_low_display.columns:
                df_low_display["SKU"] = df_low_display["SKU"].astype(str)

            # 特殊处理：Velocity 为0、无限大、空值或无效值用 '-' 替换
            def clean_velocity(x):
                if pd.isna(x) or x == 0 or x == float('inf') or x == float('-inf'):
                    return '-'
                return x

            df_low_display["Velocity"] = df_low_display["Velocity"].apply(clean_velocity)

            # Total Retail, Total Inventory, Profit 列为0的值用 '-' 替换
            df_low_display["Total Retail"] = df_low_display["Total Retail"].apply(lambda x: '-' if x == 0 else x)
            df_low_display["Total Inventory"] = df_low_display["Total Inventory"].apply(lambda x: '-' if x == 0 else x)
            df_low_display["Profit"] = df_low_display["Profit"].apply(lambda x: '-' if x == 0 else x)

            # 其他空值用字符 '-' 替换
            for col in display_columns:
                if col in df_low_display.columns:
                    if col not in ["Total Retail", "Total Inventory", "Profit", "Velocity"]:  # 这些列已经特殊处理过
                        df_low_display[col] = df_low_display[col].fillna('-')

            column_config = {
                'Item Name': st.column_config.Column(width=150),
                'Item Variation Name': st.column_config.Column(width=50),

                'SKU': st.column_config.Column(width=100),
                'Current Quantity': st.column_config.Column(width=110),
                'Total Inventory': st.column_config.Column(width=100),
                'Total Retail': st.column_config.Column(width=80),
                'Profit': st.column_config.Column(width=50),
                'Profit Margin': st.column_config.Column(width=90),
                'Velocity': st.column_config.Column(width=60),
            }

            st.dataframe(
                df_low_display[display_columns],
                column_config=column_config,
                use_container_width=False
            )
        else:
            st.info("No matching low stock items.")
    else:
        st.success("No low stock items.")

    st.markdown("---")