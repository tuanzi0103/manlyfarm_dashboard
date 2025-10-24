import streamlit as st

import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from services.analytics import (
    member_flagged_transactions,
    member_frequency_stats,
    non_member_overview,
    category_counts,
    heatmap_pivot,
    top_categories_for_customer,
    recommend_similar_categories,
    ltv_timeseries_for_customer,
    recommend_bundles_for_customer,
    churn_signals_for_member,
)


def format_phone_number(phone):
    """
    格式化手机号：移除61之前的所有字符，确保以61开头
    """
    if pd.isna(phone) or phone is None:
        return ""

    phone_str = str(phone).strip()

    # 移除所有非数字字符
    digits_only = re.sub(r'\D', '', phone_str)

    # 查找61的位置
    if '61' in digits_only:
        # 找到61第一次出现的位置
        start_index = digits_only.find('61')
        # 返回从61开始的部分
        formatted = digits_only[start_index:]

        # 确保长度合理（手机号通常10-12位）
        if len(formatted) >= 10 and len(formatted) <= 12:
            return formatted
        else:
            # 如果长度不合适，返回原始数字
            return digits_only
    else:
        # 如果没有61，返回原始数字
        return digits_only


def persisting_multiselect(label, options, key, default=None, width_chars=None, format_func=None):
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

    # 确保所有选项都是字符串类型
    options = [str(opt) for opt in options]

    # 确保默认值也是字符串类型
    default_values = [str(val) for val in st.session_state[key]]

    # 创建一个安全的 format_func，确保返回字符串
    def safe_format_func(x):
        result = format_func(x) if format_func else x
        return str(result)

    if format_func:
        return st.multiselect(label, options, default=default_values, key=key, format_func=safe_format_func)
    else:
        return st.multiselect(label, options, default=default_values, key=key)

def is_phone_number(name):
    """
    判断字符串是否为手机号（包含数字和特定字符）
    """
    if pd.isna(name) or name is None:
        return False

    name_str = str(name).strip()

    # 如果字符串只包含数字、空格、括号、加号、连字符，则认为是手机号
    if re.match(r'^[\d\s\(\)\+\-]+$', name_str):
        return True

    # 如果字符串长度在8-15之间且主要包含数字，也认为是手机号
    if 8 <= len(name_str) <= 15 and sum(c.isdigit() for c in name_str) >= 7:
        return True

    return False


def show_customer_segmentation(tx, members):
    # === 全局样式：参考 inventory 的样式设置 ===
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

    /* 统一多选框和输入框的垂直对齐 */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        align-items: start !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:24px; font-weight:700;'>👥 Customer Segmentation & Personalization</h2>",
                unsafe_allow_html=True)

    if tx.empty:
        st.info("No transaction data available.")
        return

    # always use latest uploaded data
    tx = tx.copy()
    members = members.copy()

    # === Restrict analysis to last full week (Mon–Sun before today) ===
    today = pd.Timestamp.today().normalize()
    last_sunday = today - pd.to_timedelta(today.weekday() + 1, "D")
    last_monday = last_sunday - pd.Timedelta(days=6)
    tx["Datetime"] = pd.to_datetime(tx.get("Datetime", pd.NaT), errors="coerce")
    tx = tx[(tx["Datetime"] >= last_monday) & (tx["Datetime"] <= last_sunday)]

    # --- 给交易数据打上 is_member 标记
    df = member_flagged_transactions(tx, members)

    # =========================
    # 👑 前置功能（User Analysis 之前）
    # =========================

    st.markdown("<h3 style='font-size:20px; font-weight:700;'>✨ Overview add-ons</h3>",
                unsafe_allow_html=True)

    # [1] KPI - 参考 Inventory Summary 格式
    net_col = "Net Sales" if "Net Sales" in df.columns else None
    cid_col = "Customer ID" if "Customer ID" in df.columns else None
    avg_spend_member = avg_spend_non_member = None
    if net_col and cid_col and "is_member" in df.columns:
        nets = pd.to_numeric(df[net_col], errors="coerce")
        df_kpi = df.assign(_net=nets)
        avg_spend_member = df_kpi[df_kpi["is_member"]]["_net"].mean()
        avg_spend_non_member = df_kpi[~df_kpi["is_member"]]["_net"].mean()

    # 创建类似 Inventory Summary 格式的数据框
    summary_table_data = {
        'Metric': ['Avg Spend (Enrolled)', 'Avg Spend (Not Enrolled)'],
        'Value': [
            "-" if pd.isna(avg_spend_member) else f"${avg_spend_member:,.2f}",
            "-" if pd.isna(avg_spend_non_member) else f"${avg_spend_non_member:,.2f}"
        ]
    }

    df_summary = pd.DataFrame(summary_table_data)

    # 设置列配置 - 参考 inventory 格式
    column_config = {
        'Metric': st.column_config.Column(width=150),
        'Value': st.column_config.Column(width=50),
    }

    # 显示表格
    st.dataframe(
        df_summary,
        column_config=column_config,
        hide_index=True,
        use_container_width=False
    )

    st.divider()

    # [2] 两个柱状预测 - 放在同一行
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>📊 Customer Behavior Predictions</h3>",
                unsafe_allow_html=True)

    # 使用两列布局将两个预测图表放在同一行
    col1, col2 = st.columns(2)

    time_col = next((c for c in ["Datetime", "Date", "date", "Transaction Time"] if c in df.columns), None)
    if time_col:
        with col1:
            t = pd.to_datetime(df[time_col], errors="coerce")
            day_df = df.assign(_dow=t.dt.day_name())
            dow_counts = day_df.dropna(subset=["_dow"]).groupby("_dow").size().reset_index(
                name="Predicted Transactions")
            cat_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts["_dow"] = pd.Categorical(dow_counts["_dow"], categories=cat_order, ordered=True)

            fig_dow = px.bar(
                dow_counts.sort_values("_dow"),
                x="_dow",
                y="Predicted Transactions",
                title="Shopping Days Prediction"
            )
            fig_dow.update_layout(
                width=400,
                height=400,
                xaxis_title=None,  # 去掉横轴标题
                yaxis_title="Predicted Transactions",
                margin=dict(l=40, r=10, t=60, b=30)
            )
            st.plotly_chart(fig_dow, use_container_width=False)

    # 修改：使用分类而不是具体商品名称
    category_col = next((c for c in ["Category", "Item Category", "Product Category"] if c in df.columns), None)
    qty_col = "Qty" if "Qty" in df.columns else None
    if category_col:
        with col2:
            if qty_col:
                top_categories = df.groupby(category_col)[qty_col].sum().reset_index().sort_values(qty_col,
                                                                                                   ascending=False).head(
                    15)
                # 设置柱形图宽度为更紧凑
                fig_categories = px.bar(top_categories, x=category_col, y=qty_col,
                                        title="Top Categories Prediction (Top 15)")
                fig_categories.update_layout(width=400, height=400)  # 设置图表宽度和高度
                st.plotly_chart(fig_categories, use_container_width=False)
            else:
                top_categories = df[category_col].value_counts().reset_index().rename(
                    columns={"index": "Category", category_col: "Count"}).head(15)
                # 设置柱形图宽度为更紧凑
                fig_categories = px.bar(top_categories, x="Category", y="Count",
                                        title="Top Categories Prediction (Top 15)")
                fig_categories.update_layout(width=400, height=400)  # 设置图表宽度和高度
                st.plotly_chart(fig_categories, use_container_width=False)
    else:
        # 如果没有分类列，使用商品名称但只显示大类（通过截取或分组）
        item_col = next((c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in df.columns), None)
        if item_col:
            with col2:
                # 尝试从商品名称中提取分类（取第一个单词或特定分隔符前的部分）
                df_with_category = df.copy()
                # 简单的分类提取：取第一个单词或特定分隔符前的部分
                df_with_category['_category'] = df_with_category[item_col].astype(str).str.split().str[0]

                if qty_col:
                    top_categories = df_with_category.groupby('_category')[qty_col].sum().reset_index().sort_values(
                        qty_col, ascending=False).head(15)
                    fig_categories = px.bar(top_categories, x='_category', y=qty_col,
                                            title="Top Categories Prediction (Top 15)")
                    fig_categories.update_layout(width=400, height=400)
                    st.plotly_chart(fig_categories, use_container_width=False)
                else:
                    top_categories = df_with_category['_category'].value_counts().reset_index().rename(
                        columns={"index": "Category", '_category': "Count"}).head(15)
                    fig_categories = px.bar(top_categories, x="Category", y="Count",
                                            title="Top Categories Prediction (Top 15)")
                    fig_categories.update_layout(width=400, height=400)
                    st.plotly_chart(fig_categories, use_container_width=False)

    st.divider()

    # [3] Top20 churn 风险（基于 Customer Name 计算）
    if time_col and "Customer Name" in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["_ts"] = t
        today = pd.Timestamp.today()
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - pd.Timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        # === 按 Customer Name 统计访问频率 ===
        base = df.dropna(subset=["Customer Name"])
        month_key = df["_ts"].dt.to_period("M").rename("month")
        day_key = df["_ts"].dt.date.rename("day")

        per_day = base.groupby(["Customer Name", month_key, day_key]).size().reset_index(name="visits_per_day")
        per_month = per_day.groupby(["Customer Name", "month"]).size().reset_index(name="visits")

        per_month["month_start"] = per_month["month"].dt.to_timestamp()
        mask_last = (per_month["month_start"] >= last_month_start) & (per_month["month_start"] <= last_month_end)
        pm_last = per_month.loc[mask_last, ["Customer Name", "visits"]].rename(columns={"visits": "Last Month Visit"})
        hist_avg = per_month.loc[~mask_last].groupby("Customer Name")["visits"].mean().reset_index(
            name="Average Visit")

        churn_tag = hist_avg.merge(pm_last, on="Customer Name", how="left").fillna({"Last Month Visit": 0})

        # ✅ 过滤掉原本就偶尔来的顾客，保留常客
        churn_tag = churn_tag[churn_tag["Average Visit"] >= 2]

        # ✅ 过滤掉 Customer Name 是手机号的记录
        churn_tag = churn_tag[~churn_tag["Customer Name"].apply(is_phone_number)]

        churn_tag = churn_tag.sort_values("Average Visit", ascending=False).head(20)

        # 映射手机号（如果 members 表有）
        if "Square Customer ID" in members.columns:
            id_name = tx[["Customer ID", "Customer Name"]].drop_duplicates().dropna(subset=["Customer ID"])
            phones_map = (
                members.rename(columns={"Square Customer ID": "Customer ID", "Phone Number": "Phone"})
                [["Customer ID", "Phone"]]
                .dropna(subset=["Customer ID"])
                .drop_duplicates("Customer ID")
            )
            phones_map["Customer ID"] = phones_map["Customer ID"].astype(str)
            phones_map["Phone"] = phones_map["Phone"].apply(format_phone_number)
            churn_tag = churn_tag.merge(id_name, on="Customer Name", how="left").merge(
                phones_map, on="Customer ID", how="left"
            )

        st.markdown("<h3 style='font-size:20px; font-weight:700;'>Top 20 Regulars who didn't come last month</h3>",
                    unsafe_allow_html=True)

        # === 修改：设置表格列宽配置 ===
        column_config = {
            'Customer Name': st.column_config.Column(width=105),
            'Customer ID': st.column_config.Column(width=100),
            'Phone': st.column_config.Column(width=90),
            'Average Visit': st.column_config.Column(width=90),
            'Last Month Visit': st.column_config.Column(width=110),
        }

        st.dataframe(
            churn_tag[["Customer Name", "Customer ID", "Phone",
                       "Average Visit", "Last Month Visit"]],
            column_config=column_config,
            use_container_width=False
        )

    st.divider()

    # [4] 姓名/ID 搜索（显示姓名，支持用 ID 搜索）
    options = []
    if "Customer ID" in tx.columns and "Customer Name" in tx.columns:
        options = (tx[["Customer ID", "Customer Name"]]
                   .dropna(subset=["Customer ID"])
                   .drop_duplicates("Customer ID"))
        # 🚩 确保 Customer ID 全部是字符串，避免 multiselect 报错
        options["Customer ID"] = options["Customer ID"].astype(str)
        options = options.to_dict(orient="records")

    # 🔹 使用三列布局缩短下拉框宽度，与 inventory.py 保持一致
    col_search, _ = st.columns([1, 6])
    with col_search:
        # 创建选项映射
        # ✅ 下拉框只显示用户名，不显示ID
        option_dict = {str(opt["Customer ID"]): str(opt["Customer Name"]) for opt in options}

        # 确保选项是字符串类型
        customer_options = [str(opt["Customer ID"]) for opt in options]

        sel_ids = persisting_multiselect(
            "🔎 Search customers",
            options=customer_options,
            format_func=lambda x: option_dict.get(x, x),
            key="customer_search",
            width_chars=15
        )

    if sel_ids:
        chosen = tx[tx["Customer ID"].astype(str).isin(sel_ids)]
        st.subheader("All transactions for selected customers")

        column_config = {
            "Datetime": st.column_config.Column(width=120),
            "Customer Name": st.column_config.Column(width=120),
            "Customer ID": st.column_config.Column(width=140),
            "Category": st.column_config.Column(width=140),
            "Item": st.column_config.Column(width=110),
            "Qty": st.column_config.Column(width=40),
            "Net Sales": st.column_config.Column(width=80),
        }

        # ✅ 仅显示指定列（按顺序）
        display_cols = ["Datetime", "Customer Name", "Customer ID", "Category", "Item", "Qty", "Net Sales"]
        existing_cols = [c for c in display_cols if c in chosen.columns]

        st.dataframe(
            chosen[existing_cols],
            column_config=column_config,
            use_container_width=False,  # ✅ 关闭容器自适应，列宽才生效
            hide_index=True
        )

        if qty_col:
            # 修改：使用分类而不是具体商品
            category_col_display = next(
                (c for c in ["Category", "Item Category", "Product Category"] if c in chosen.columns), None)

            if category_col_display:
                top5 = (chosen.groupby(["Customer ID", "Customer Name", category_col_display])[qty_col].sum()
                        .reset_index()
                        .sort_values(["Customer Name", qty_col], ascending=[True, False])
                        .groupby("Customer ID").head(5))

                st.markdown(
                    "<h3 style='font-size:20px; font-weight:700;'>Frequently purchased categories (Top 5 / customer)</h3>",
                    unsafe_allow_html=True)

                # === 修改：设置表格列宽配置，使用分类列 ===
                column_config = {
                    'Customer ID': st.column_config.Column(width=150),
                    'Customer Name': st.column_config.Column(width=110),
                    category_col_display: st.column_config.Column(width=160),
                    qty_col: st.column_config.Column(width=40),
                }

                st.dataframe(top5, column_config=column_config, use_container_width=False)

            else:
                # 如果没有分类列，使用商品名称但显示为分类
                item_col_display = next(
                    (c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in chosen.columns), None)
                if item_col_display:
                    # 从商品名称中提取分类
                    chosen_with_category = chosen.copy()
                    chosen_with_category['_category'] = \
                        chosen_with_category[item_col_display].astype(str).str.split().str[0]

                    top5 = (chosen_with_category.groupby(["Customer ID", "Customer Name", '_category'])[qty_col].sum()
                            .reset_index()
                            .sort_values(["Customer Name", qty_col], ascending=[True, False])
                            .groupby("Customer ID").head(5))

                    st.markdown(
                        "<h3 style='font-size:20px; font-weight:700;'>Frequently purchased categories (Top 5 / customer)</h3>",
                        unsafe_allow_html=True)

                    # === 修改：设置表格列宽配置，使用分类列 ===
                    column_config = {
                        'Customer ID': st.column_config.Column(width=150),
                        'Customer Name': st.column_config.Column(width=110),
                        '_category': st.column_config.Column(width=160, title="Category"),
                        qty_col: st.column_config.Column(width=40),
                    }

                    st.dataframe(top5, column_config=column_config, use_container_width=False)

    st.divider()

    # [5] Heatmap 可切换
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>Heatmap (selectable metric)</h3>",
                unsafe_allow_html=True)

    # 🔹 使用三列布局缩短下拉框宽度，与 inventory.py 保持一致
    col_metric, _ = st.columns([1, 6])
    with col_metric:
        # === 修改：设置选择框宽度 ===
        st.markdown("""
        <style>
        div[data-testid*="stSelectbox"][aria-label="Metric"],
        div[data-testid*="stSelectbox"][data-baseweb="select"][aria-label="Metric"] {
            width: 15ch !important;
            min-width: 15ch !important;
            max-width: 15ch !important;
        }
        </style>
        """, unsafe_allow_html=True)

        metric = st.selectbox("Metric", ["net sales", "number of transactions"], index=0, key="heatmap_metric")

    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        base = df.assign(_date=t)
        base["_hour"] = base["_date"].dt.hour
        base["_dow"] = base["_date"].dt.day_name()
        if metric == "net sales" and net_col:
            agg = base.groupby(["_dow", "_hour"])[net_col].sum().reset_index(name="value")
        else:
            txn_col2 = "Transaction ID" if "Transaction ID" in base.columns else None
            if txn_col2:
                agg = base.groupby(["_dow", "_hour"])[txn_col2].nunique().reset_index(name="value")
            else:
                agg = base.groupby(["_dow", "_hour"]).size().reset_index(name="value")
        pv = agg.pivot(index="_dow", columns="_hour", values="value").fillna(0)

        # === 修改：设置热力图宽度 ===
        fig_heatmap = px.imshow(pv, aspect="auto", title=f"Heatmap by {metric.title()} (Hour x Day)")
        fig_heatmap.update_layout(width=600)  # 设置图表宽度
        st.plotly_chart(fig_heatmap, use_container_width=False)