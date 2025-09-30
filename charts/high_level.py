import streamlit as st
import pandas as pd
import plotly.express as px


def persisting_multiselect(label, options, key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)


def _safe_sum(df, col):
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df.get(col), errors="coerce").fillna(0).sum())


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    if tx.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    today = pd.Timestamp.today().normalize()
    tx["date"] = pd.to_datetime(tx["Datetime"]).dt.normalize()

    # === 顶部 KPI ===
    latest_date = tx["date"].max()
    df_latest = tx[tx["date"] == latest_date]

    kpis = {
        "Daily Net Sales": df_latest["Net Sales"].sum(),
        "Daily Transactions": len(df_latest),
        "Avg Transaction": df_latest["Net Sales"].mean(),
        "3M Avg": tx.groupby("date")["Net Sales"].sum().rolling(90, min_periods=1).mean().iloc[-1],
        "6M Avg": tx.groupby("date")["Net Sales"].sum().rolling(180, min_periods=1).mean().iloc[-1],
        "Inventory Value": _safe_sum(inv, "Current Quantity Vie Market & Bar"),
        "Profit (Amount)": df_latest["Gross Sales"].sum() - df_latest["Net Sales"].sum(),
        "Items Sold": df_latest["Qty"].sum() if "Qty" in df_latest.columns else 0,
    }

    # ✅ KPI 日期格式化
    st.markdown(f"### 📅 Latest available date: {latest_date.strftime('%Y-%m-%d')}")

    labels = list(kpis.keys())
    values = list(kpis.values())
    for row in range(0, len(labels), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row + i
            if idx < len(labels):
                label = labels[idx]
                value = values[idx]
                display = "-" if pd.isna(value) else f"{value:,.2f}"
                with col:
                    st.markdown(
                        f"<div style='font-size:28px; font-weight:600'>{display}</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(label)

    st.markdown("---")

    # === 三个多选框 ===
    st.subheader("🔍 Select Parameters")

    time_range = persisting_multiselect("Choose time range", ["Custom dates", "WTD", "MTD", "YTD"], key="hl_time")
    data_options = [
        "Daily Net Sales", "Daily Transactions", "Avg Transaction",
        "3M Avg", "6M Avg",
        "Inventory Value", "Profit (Amount)", "Items Sold"
    ]
    data_sel = persisting_multiselect("Choose data type", data_options, key="hl_data")

    all_cats = sorted(tx["Category"].fillna("Unknown").unique().tolist())
    bar_cats = ["Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"]
    cats_sel = persisting_multiselect("Choose categories", all_cats + ["bar", "retail"], key="hl_cats")

    if time_range and data_sel and cats_sel:
        df = tx.copy()
        df["date"] = pd.to_datetime(df["Datetime"]).dt.normalize()

        # === 时间过滤（往前推） ===
        if "WTD" in time_range:
            start = today - pd.Timedelta(days=7)
            df = df[df["date"] >= start]

        if "MTD" in time_range:
            start = today - pd.Timedelta(days=30)
            df = df[df["date"] >= start]

        if "YTD" in time_range:
            start = today - pd.Timedelta(days=365)
            df = df[df["date"] >= start]

        if "Custom dates" in time_range:
            t1 = st.date_input("From")
            t2 = st.date_input("To")
            if t1 and t2:
                df = df[(df["date"] >= pd.Timestamp(t1)) & (df["date"] <= pd.Timestamp(t2))]

        # === bar/retail 分组 ===
        df["cat_group"] = df["Category"].fillna("Unknown")
        df_bar = df[df["Category"].isin(bar_cats)].copy()
        df_bar["cat_group"] = "bar"
        df_retail = df[~df["Category"].isin(bar_cats)].copy()
        df_retail["cat_group"] = "retail"
        df = pd.concat([df, df_bar, df_retail], ignore_index=True)

        # === 聚合 ===
        grouped = df.groupby(["date", "cat_group"]).agg(
            net_sales=("Net Sales", "sum"),
            transactions=("Datetime", "count"),
            avg_txn=("Net Sales", "mean"),
            gross=("Gross Sales", "sum"),
            qty=("Qty", "sum")
        ).reset_index()
        grouped["profit"] = grouped["gross"] - grouped["net_sales"]
        grouped["3M_Rolling"] = grouped.groupby("cat_group")["net_sales"].transform(
            lambda x: x.rolling(90, min_periods=1).mean()
        )
        grouped["6M_Rolling"] = grouped.groupby("cat_group")["net_sales"].transform(
            lambda x: x.rolling(180, min_periods=1).mean()
        )
        grouped["inventory_value"] = _safe_sum(inv, "Current Quantity Vie Market & Bar")

        # ✅ 按 category 过滤
        grouped = grouped[grouped["cat_group"].isin(cats_sel)]

        # ✅ 格式化日期（图表 & 表格统一 YYYY-MM-DD）
        grouped["date"] = grouped["date"].dt.strftime("%Y-%m-%d")

        # === 图表 ===
        mapping = {
            "Daily Net Sales": ("net_sales", "Daily Net Sales"),
            "Daily Transactions": ("transactions", "Daily Transactions"),
            "Avg Transaction": ("avg_txn", "Avg Transaction"),
            "3M Avg": ("3M_Rolling", "3M Avg"),
            "6M Avg": ("6M_Rolling", "6M Avg"),
            "Inventory Value": ("inventory_value", "Inventory Value"),
            "Profit (Amount)": ("profit", "Profit (Amount)"),
            "Items Sold": ("qty", "Items Sold"),
        }

        for metric in data_sel:
            if metric not in mapping:
                continue
            y, colname = mapping[metric]

            plot_df = grouped.dropna(subset=[y])
            if len(plot_df["date"].unique()) > 1:
                fig = px.line(
                    plot_df, x="date", y=y, color="cat_group",
                    title=f"{colname} by Category", markers=True
                )
            else:
                fig = px.scatter(
                    plot_df, x="date", y=y, color="cat_group",
                    title=f"{colname} by Category", size_max=12
                )

            # ✅ 横坐标日期格式化
            fig.update_layout(xaxis=dict(type="category"))

            st.plotly_chart(fig, use_container_width=True)

        # === 表格 ===
        st.subheader("📋 Detailed Data")
        cols_to_show = ["date", "cat_group"]
        for sel in data_sel:
            if sel in mapping:
                cols_to_show.append(mapping[sel][0])

        st.dataframe(grouped[cols_to_show], use_container_width=True)

    else:
        st.info("Please select time range, data, and category to generate the chart.")
