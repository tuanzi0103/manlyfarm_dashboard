import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from services.simulator import simulate_transactions   # 🔹 引入模拟器

def persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)

def _safe_sum(df, col):
    return float(pd.to_numeric(df.get(col), errors="coerce").fillna(0).sum())

def _inventory_total(inv: pd.DataFrame) -> float:
    if inv is None or inv.empty:
        return 0.0
    df = inv.copy()
    qty_col = None
    for c in df.columns:
        if str(c).strip().lower().startswith("current quantity"):
            qty_col = c
            break
    if qty_col is None:
        return 0.0
    qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).abs()
    for eq in ["Stock-by Equivalent", "Sell-by Equivalent"]:
        if eq in df.columns:
            eqv = pd.to_numeric(df[eq], errors="coerce").fillna(1.0)
            return float((qty * eqv).sum())
    return float(qty.sum())

def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    st.header("🧾 Sales Report by Category")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    with st.sidebar:
        st.subheader("Filter")
        all_cats = sorted(tx.get("Category", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        sel = persisting_multiselect("Categories", all_cats, key="sr_cats")

    df = tx.copy()
    df["Qty"] = pd.to_numeric(df.get("Qty"), errors="coerce").fillna(0).abs()
    df["Net Sales"] = pd.to_numeric(df.get("Net Sales"), errors="coerce").fillna(0.0)

    if sel:
        df = df[df["Category"].astype(str).isin(sel)]

    # Items sold & Net value
    g = df.groupby("Category", as_index=False).agg(items_sold=("Qty", "sum"),
                                                   net_value=("Net Sales", "sum")).sort_values("items_sold", ascending=False)
    if not g.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(g, x="Category", y="items_sold", title="Items Sold (by Category)"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(g, x="Category", y="net_value", title="Net Sales (by Category)"), use_container_width=True)
        st.dataframe(g, use_container_width=True)
    else:
        st.info("No data under current filters.")

    # Weekly sales WoW
    st.subheader("Weekly Sales vs Previous Week (WoW)")

    # 🔹 模拟数据控制
    option = st.selectbox("Select synthetic data period", ["1M", "3M", "6M", "9M"], key="sr_sim_option")
    col1, col2 = st.columns(2)
    if col1.button("Generate WoW Data"):
        months = int(option.replace("M", ""))
        st.session_state["simulated_sales_tx"] = simulate_transactions(tx, months=months)
    if col2.button("❌ Clear WoW Data"):
        if "simulated_sales_tx" in st.session_state:
            del st.session_state["simulated_sales_tx"]

    # 使用模拟数据或真实数据
    if "simulated_sales_tx" in st.session_state:
        df = st.session_state["simulated_sales_tx"]

    wdf = df.copy()
    wdf["week"] = wdf["Datetime"].dt.to_period("W").apply(lambda r: r.start_time)
    wow = wdf.groupby("week", as_index=False)["Net Sales"].sum().sort_values("week")
    if len(wow) >= 2:
        wow["prev"] = wow["Net Sales"].shift(1)
        wow["wow_change"] = (wow["Net Sales"] - wow["prev"]).fillna(0.0)
        st.plotly_chart(
            px.bar(
                wow.dropna(subset=["prev"]),
                x="week",
                y="wow_change",
                title="Week-over-Week Change (Net Sales)"
            ),
            use_container_width=True
        )
    else:
        st.info("Not enough weekly points to compute WoW.")

    # Totals
    net_total = _safe_sum(tx, "Net Sales")
    inv_total = _inventory_total(inv)
    comp = pd.DataFrame({"metric": ["Total Net Sales", "Total Inventory"], "value": [net_total, inv_total]})
    st.plotly_chart(px.bar(comp, x="metric", y="value", title="Totals Comparison"), use_container_width=True)
