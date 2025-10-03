import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)


def _safe_sum(df, col):
    return float(pd.to_numeric(df.get(col), errors="coerce").fillna(0).sum())


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    st.header("🧾 Sales Report by Category")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    # 🔹 确保 Datetime 是时间类型
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")

    # ---------------- Time Range Filter ----------------
    st.subheader("📅 Time Range")
    range_opt = st.selectbox("Select range", ["Custom dates", "WTD", "MTD", "YTD"], key="sr_range")

    today = pd.Timestamp.today().normalize()
    start_date, end_date = None, today

    if range_opt == "Custom dates":
        t1 = st.date_input("From", today - timedelta(days=7))
        t2 = st.date_input("To", today)
        if t1 and t2:
            start_date, end_date = pd.to_datetime(t1), pd.to_datetime(t2)
    elif range_opt == "WTD":
        start_date = today - timedelta(days=7)
    elif range_opt == "MTD":
        start_date = today - timedelta(days=30)
    elif range_opt == "YTD":
        start_date = today - timedelta(days=365)

    if start_date is not None and end_date is not None:
        mask = (tx["Datetime"] >= pd.to_datetime(start_date)) & (tx["Datetime"] <= pd.to_datetime(end_date))
        tx = tx.loc[mask]

    # ---------------- Bar Charts ----------------
    df = tx.copy()
    df["Qty"] = pd.to_numeric(df.get("Qty"), errors="coerce").fillna(0).abs()
    df["Net Sales"] = pd.to_numeric(df.get("Net Sales"), errors="coerce").fillna(0.0)

    g = df.groupby("Category", as_index=False).agg(
        items_sold=("Qty", "sum"),
        net_value=("Net Sales", "sum")
    ).sort_values("items_sold", ascending=False)

    if not g.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(g, x="Category", y="items_sold", title="Items Sold (by Category)"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(g, x="Category", y="net_value", title="Net Sales (by Category)"),
                            use_container_width=True)
    else:
        st.info("No data under current filters.")
        return

    # ---------------- Group definitions ----------------
    bar_cats = ["Cafe Drinks", "Smoothie Bar", "Soups", "Sweet Treats", "Wraps & Salads"]
    retail_cats = [c for c in df["Category"].unique() if c not in bar_cats]

    # helper: compute weekly summary
    def weekly_summary(data, cats):
        sub = data[data["Category"].isin(cats)].copy()
        if sub.empty:
            return pd.DataFrame()

        sub["week"] = sub["Datetime"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly = sub.groupby(["week", "Category"], as_index=False).agg(
            items_sold=("Qty", "sum"),
            net_sales=("Net Sales", "sum")
        )
        if weekly.empty:
            return pd.DataFrame()

        latest_week = weekly["week"].max()
        weeks_sorted = weekly["week"].sort_values().unique()
        prev_week = weeks_sorted[-2] if len(weeks_sorted) > 1 else None

        curr = weekly[weekly["week"] == latest_week].set_index("Category")
        prev = weekly[weekly["week"] == prev_week].set_index("Category") if prev_week is not None else pd.DataFrame()

        result = curr.copy()
        result["prior_week"] = prev["net_sales"] if not prev.empty else 0

        # ✅ Weekly change (环比增长率，带阈值和N/A处理)
        MIN_BASE = 50
        result["weekly_change"] = np.where(
            result["prior_week"] > MIN_BASE,
            (result["net_sales"] - result["prior_week"]) / result["prior_week"],
            np.nan
        )

        result["per_day"] = result["items_sold"] / 7
        return result.reset_index()

    # helper: 格式化 + 高亮
    def format_change(x):
        if pd.isna(x):
            return "N/A"
        return f"{x*100:+.2f}%"

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
    st.subheader("📊 Bar Categories")
    bar_df = weekly_summary(df, bar_cats)
    if not bar_df.empty:
        bar_df = bar_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "net_sales": "Sum of Net Sales",
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })
        bar_df["Weekly change"] = bar_df["Weekly change"].apply(format_change)

        st.dataframe(
            bar_df[["Row Labels", "Sum of Items Sold", "Sum of Net Sales", "Weekly change", "Per day"]]
            .style.applymap(highlight_change, subset=["Weekly change"]),
            use_container_width=True
        )
    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table + Multiselect ----------------
    st.subheader("📊 Retail Categories")
    all_retail_cats = sorted(df[df["Category"].isin(retail_cats)]["Category"].dropna().unique().tolist())
    sel_retail_cats = persisting_multiselect("Select Retail Categories", all_retail_cats, key="sr_retail_cats")

    retail_df = weekly_summary(df, retail_cats)
    if not retail_df.empty:
        retail_df = retail_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "net_sales": "Sum of Net Sales",
            "weekly_change": "Weekly change"
        })
        if sel_retail_cats:
            retail_df = retail_df[retail_df["Row Labels"].isin(sel_retail_cats)]
        retail_df["Weekly change"] = retail_df["Weekly change"].apply(format_change)

        st.dataframe(
            retail_df[["Row Labels", "Sum of Items Sold", "Sum of Net Sales", "Weekly change"]]
            .style.applymap(highlight_change, subset=["Weekly change"]),
            use_container_width=True
        )
    else:
        st.info("No data for Retail categories.")

    # ---------------- Comment (Retail Top3 Categories) ----------------
    st.markdown("### 💬 Comment")
    if not df[df["Category"].isin(retail_cats)].empty:
        retail_cats_summary = (df[df["Category"].isin(retail_cats)]
                               .groupby("Category")["Net Sales"]
                               .sum()
                               .reset_index()
                               .sort_values("Net Sales", ascending=False)
                               .head(9))
        lines = []
        for i in range(0, len(retail_cats_summary), 3):
            chunk = retail_cats_summary.iloc[i:i+3]
            line = " ".join([f"${int(v)} {n}" for n, v in zip(chunk["Category"], chunk["Net Sales"])])
            lines.append(line)
        st.text("\n".join(lines))
    else:
        st.info("No retail categories available for comments.")
