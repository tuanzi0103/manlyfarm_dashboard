import streamlit as st
import plotly.express as px
import pandas as pd
from services.simulator import simulate_transactions

def persisting_multiselect(label, options, key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)

def _safe_sum(df, col):
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df.get(col), errors="coerce").fillna(0).sum())

# === 缓存聚合逻辑 ===
@st.cache_data(show_spinner=False)
def compute_trend(tx: pd.DataFrame, total_inv: float):
    trend = tx.copy()
    trend["date"] = trend["Datetime"].dt.floor("D")
    trend_stats = trend.groupby("date").agg(
        net_sales=("Net Sales", "sum"),
        transactions=("Datetime", "count"),
        avg_txn=("Net Sales", "mean"),
        gross=("Gross Sales", "sum")
    ).reset_index()
    trend_stats["profit"] = trend_stats["gross"] - trend_stats["net_sales"]
    trend_stats["inventory_value"] = total_inv
    return trend_stats

@st.cache_data(show_spinner=False)
def compute_daily_grouped(df: pd.DataFrame):
    df = df.copy()
    df["date"] = df["Datetime"].dt.floor("D")
    return df.groupby(["date", "category_clean"], as_index=False).agg(
        net_sales=("Net Sales", "sum")
    )

def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("🏁 High Level Report")

    # ===== 时间范围 =====
    st.subheader("⏱ Time Range")
    option = st.selectbox("Select range", ["All", "Custom", "WTD", "MTD", "YTD"], key="time_range")
    today = pd.Timestamp.today().normalize()

    # ===== Data Simulation =====
    st.subheader("🧪 Data Simulation")
    option_sim = st.selectbox("Synthetic data period", ["1M", "3M", "6M", "9M"], key="sim_option")
    col1, col2 = st.columns(2)
    if col1.button("Generate Data"):
        months = int(option_sim.replace("M", ""))
        st.session_state["simulated_tx"] = simulate_transactions(pd.DataFrame(), months=months)
    if col2.button("❌ Clear Data"):
        if "simulated_tx" in st.session_state:
            del st.session_state["simulated_tx"]

    # 使用模拟数据（如果有）
    if "simulated_tx" in st.session_state:
        tx = st.session_state["simulated_tx"]

    # ===== 如果还是没数据，显示默认提示并停止 =====
    if tx is None or tx.empty:
        st.warning("No transaction data available. Use 'Generate Data' above to create synthetic transactions.")
        persisting_multiselect("Choose categories", ["retail", "bar", "other"],
                               key="hl_cats", default=["retail", "bar", "other"])
        return

    # ===== 应用时间范围过滤 =====
    if option == "Custom":
        time_from = st.date_input("From")
        time_to = st.date_input("To")
        if time_from and time_to:
            tx = tx[(tx["Datetime"].dt.date >= time_from) & (tx["Datetime"].dt.date <= time_to)]
    elif option == "WTD":
        monday = today - pd.Timedelta(days=today.weekday())
        tx = tx[tx["Datetime"].dt.date >= monday.date()]
    elif option == "MTD":
        first_day = today.replace(day=1)
        tx = tx[tx["Datetime"].dt.date >= first_day.date()]
    elif option == "YTD":
        first_day = today.replace(month=1, day=1)
        tx = tx[tx["Datetime"].dt.date >= first_day.date()]

    # ===== 顶部 KPI (两行显示) =====
    k1, k2, k3, k4 = st.columns(4)
    k5, k6, k7, k8 = st.columns(4)

    k1.metric("Daily Net Sales", f"{_safe_sum(tx, 'Net Sales'):.2f}")
    k2.metric("Daily Transactions", len(tx))
    avg_txn = _safe_sum(tx, "Net Sales") / max(len(tx), 1)
    k3.metric("Avg Transaction", f"{avg_txn:.2f}")

    daily = tx.copy()
    daily["date"] = daily["Datetime"].dt.floor("D")
    daily_sum = daily.groupby("date", as_index=False)["Net Sales"].sum().sort_values("date")
    daily_sum["3M_Rolling"] = daily_sum["Net Sales"].rolling(90, min_periods=1).mean()
    daily_sum["6M_Rolling"] = daily_sum["Net Sales"].rolling(180, min_periods=1).mean()

    k4.metric("3M Avg", f"{daily_sum['3M_Rolling'].iloc[-1]:.2f}" if not daily_sum.empty else "N/A")
    k5.metric("6M Avg", f"{daily_sum['6M_Rolling'].iloc[-1]:.2f}" if not daily_sum.empty else "N/A")

    total_inv = _safe_sum(inv, "Current Quantity Vie Market & Bar") if inv is not None else 0
    k6.metric("Inventory Value", f"{total_inv:.2f}")

    profit = _safe_sum(tx, "Gross Sales") - _safe_sum(tx, "Net Sales")
    k7.metric("Profit (Amount)", f"{profit:.2f}")

    k8.metric("Items Sold", f"{_safe_sum(tx, 'Qty'):.0f}")

    # ===== 分类多选折线图 =====
    st.subheader("📈 Daily Net Sales by Location/Category")

    if "location" in tx.columns:
        tx["category_clean"] = tx["location"].fillna("other").astype(str)
        tx.loc[tx["category_clean"].str.contains("bar", case=False, na=False), "category_clean"] = "bar"
        tx.loc[tx["category_clean"].str.contains("retail", case=False, na=False), "category_clean"] = "retail"
        mask_known = tx["category_clean"].isin(["bar", "retail"])
        tx.loc[~mask_known, "category_clean"] = "other"
    elif "Category" in tx.columns:
        tx["category_clean"] = tx["Category"].fillna("other").astype(str)
    else:
        tx["category_clean"] = "other"

    cats = sorted(tx["category_clean"].unique().tolist())
    for d in ["retail", "bar", "other"]:
        if d not in cats:
            cats.append(d)

    sel = persisting_multiselect("Choose categories", cats, key="hl_cats", default=["retail", "bar", "other"])

    df = tx.copy()
    df["date"] = df["Datetime"].dt.floor("D")
    if sel:
        df = df[df["category_clean"].astype(str).isin(sel)]

    daily_grouped = compute_daily_grouped(df)  # ✅ 缓存
    if not daily_grouped.empty:
        fig = px.line(
            daily_grouped,
            x="date",
            y="net_sales",
            color="category_clean",
            title="Daily Net Sales (multi-select)",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected categories.")

    # ===== 融合趋势图 =====
    st.subheader("📊 Sales Trend & KPIs Over Time")
    trend_stats = compute_trend(tx, total_inv)  # ✅ 缓存

    if not trend_stats.empty:
        st.dataframe(trend_stats, use_container_width=True)
        st.plotly_chart(
            px.line(
                trend_stats,
                x="date",
                y=["net_sales", "transactions", "avg_txn", "inventory_value", "profit"],
                title="Sales & KPIs Over Time"
            ),
            use_container_width=True
        )
    else:
        st.info("No trend data available.")
