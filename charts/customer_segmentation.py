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


def show_customer_segmentation(tx, members):
    st.header("👥 Customer Segmentation & Personalization")

    if tx.empty:
        st.info("No transaction data available.")
        return

    # --- 给交易数据打上 is_member 标记
    df = member_flagged_transactions(tx, members)

    # =========================
    # 👑 前置功能（User Analysis 之前）
    # =========================

    st.markdown("### ✨ Overview add-ons")

    # [1] KPI
    net_col = "Net Sales" if "Net Sales" in df.columns else None
    cid_col = "Customer ID" if "Customer ID" in df.columns else None
    avg_spend_member = avg_spend_non_member = None
    if net_col and cid_col and "is_member" in df.columns:
        nets = pd.to_numeric(df[net_col], errors="coerce")
        df_kpi = df.assign(_net=nets)
        avg_spend_member = df_kpi[df_kpi["is_member"]]["_net"].mean()
        avg_spend_non_member = df_kpi[~df_kpi["is_member"]]["_net"].mean()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Average spend per customers **enrolled**",
                  "-" if pd.isna(avg_spend_member) else f"{avg_spend_member:,.2f}")
    with c2:
        st.metric("Average spend per customers **not enrolled**",
                  "-" if pd.isna(avg_spend_non_member) else f"{avg_spend_non_member:,.2f}")

    st.divider()

    # [2] 两个柱状预测
    time_col = next((c for c in ["Datetime", "Date", "date", "Transaction Time"] if c in df.columns), None)
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        day_df = df.assign(_dow=t.dt.day_name())
        dow_counts = day_df.dropna(subset=["_dow"]).groupby("_dow").size().reset_index(name="Predicted Transactions")
        cat_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts["_dow"] = pd.Categorical(dow_counts["_dow"], categories=cat_order, ordered=True)
        st.plotly_chart(px.bar(dow_counts.sort_values("_dow"), x="_dow", y="Predicted Transactions",
                               title="Prediction: What days customers are going to shop"), use_container_width=True)

    item_col = next((c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in df.columns), None)
    qty_col = "Qty" if "Qty" in df.columns else None
    if item_col:
        if qty_col:
            top_items = df.groupby(item_col)[qty_col].sum().reset_index().sort_values(qty_col, ascending=False).head(15)
            st.plotly_chart(px.bar(top_items, x=item_col, y=qty_col,
                                   title="Prediction: What they will buy (Top 15)"), use_container_width=True)
        else:
            top_items = df[item_col].value_counts().reset_index().rename(
                columns={"index": "Item", item_col: "Count"}).head(15)
            st.plotly_chart(px.bar(top_items, x="Item", y="Count",
                                   title="Prediction: What they will buy (Top 15)"), use_container_width=True)

    st.divider()

    # [3] Top20 churn 风险
    if time_col and cid_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["_ts"] = t
        today = pd.Timestamp.today()
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - pd.Timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        month_key = df["_ts"].dt.to_period("M").rename("month")
        day_key = df["_ts"].dt.date.rename("day")

        txn_col = "Transaction ID" if "Transaction ID" in df.columns else None
        base = df.drop_duplicates([cid_col, "_ts", txn_col]) if txn_col else df

        per_day = base.groupby([cid_col, month_key, day_key]).size().reset_index(name="visits_per_day")
        per_month = per_day.groupby([cid_col, "month"]).size().reset_index(name="visits")

        per_month["month_start"] = per_month["month"].dt.to_timestamp()
        mask_last = (per_month["month_start"] >= last_month_start) & (per_month["month_start"] <= last_month_end)
        pm_last = per_month.loc[mask_last, [cid_col, "visits"]].rename(columns={"visits": "visits_last_month"})
        hist_avg = per_month.loc[~mask_last].groupby(cid_col)["visits"].mean().reset_index(
            name="avg_visits_per_month_excl_last")

        churn_tag = hist_avg.merge(pm_last, on=cid_col, how="left").fillna({"visits_last_month": 0})
        churn_tag = churn_tag.sort_values("avg_visits_per_month_excl_last", ascending=False).head(20)

        names_map = (tx.loc[:, ["Customer ID", "Customer Name"]]
                     .dropna(subset=["Customer ID"])
                     .drop_duplicates("Customer ID"))
        names_map["Customer ID"] = names_map["Customer ID"].astype(str)

        phones_map = (members.rename(columns={"Square Customer ID": "Customer ID", "Phone Number": "Phone"})
                      [["Customer ID", "Phone"]]
                      .dropna(subset=["Customer ID"])
                      .drop_duplicates("Customer ID"))
        phones_map["Customer ID"] = phones_map["Customer ID"].astype(str)

        # ✅ 格式化手机号：移除61之前的所有字符
        phones_map["Phone"] = phones_map["Phone"].apply(format_phone_number)

        risky = (churn_tag.assign(**{"Customer ID": churn_tag["Customer ID"].astype(str)})
                 .merge(names_map, on="Customer ID", how="left")
                 .merge(phones_map, on="Customer ID", how="left"))

        st.subheader("Top 20: Regulars who **didn't come last month**")
        st.dataframe(
            risky[["Customer Name", "Customer ID", "Phone", "avg_visits_per_month_excl_last", "visits_last_month"]],
            use_container_width=True)

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

    # 使用紧凑的三列布局，与 high_level.py 保持一致
    search_col1, search_col2, search_col3 = st.columns([1, 1, 1])
    with search_col1:
        sel_ids = persisting_multiselect(
            "🔎 Search customers by name",
            options=[opt["Customer ID"] for opt in options],
            key="customer_search",
            default=[]
        )

    if sel_ids:
        chosen = tx[tx["Customer ID"].astype(str).isin(sel_ids)]
        st.subheader("All transactions for selected customers")
        st.dataframe(chosen, use_container_width=True)

        if item_col and qty_col:
            top5 = (chosen.groupby(["Customer ID", "Customer Name", item_col])[qty_col].sum()
                    .reset_index()
                    .sort_values(["Customer Name", qty_col], ascending=[True, False])
                    .groupby("Customer ID").head(5))
            st.subheader("Frequently purchased items (Top 5 / customer)")
            st.dataframe(top5, use_container_width=True)

    st.divider()

    # [5] Heatmap 可切换
    st.subheader("Heatmap (selectable metric)")

    # 使用紧凑的三列布局，与 high_level.py 保持一致
    heatmap_col1, heatmap_col2, heatmap_col3 = st.columns([1, 1, 1])
    with heatmap_col1:
        metric = st.selectbox(
            "Metric",
            ["net sales", "number of transactions"],
            index=0,
            key="heatmap_metric"
        )

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
        st.plotly_chart(px.imshow(pv, aspect="auto", title=f"Heatmap by {metric.title()} (Hour x Day)"),
                        use_container_width=True)