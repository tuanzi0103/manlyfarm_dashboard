import streamlit as st
import plotly.express as px
import pandas as pd
from services.simulator import simulate_transactions   # 🔹 引入模拟器

def persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)

def _safe_sum(df, col):
    return float(pd.to_numeric(df.get(col), errors="coerce").fillna(0).sum())

def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("🏁 High Level Report")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    # ===== 时间过滤器 =====
    st.sidebar.subheader("⏱ Time Range")
    option = st.sidebar.selectbox("Select range", ["All", "Custom", "WTD", "MTD", "YTD"])
    today = pd.Timestamp.today().normalize()

    if option == "Custom":
        time_from = st.sidebar.date_input("From")
        time_to = st.sidebar.date_input("To")
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

    # ===== 顶部 KPI =====
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    k1.metric("Daily Net Sales", f"{_safe_sum(tx, 'Net Sales'):.2f}")
    k2.metric("Daily Transactions", len(tx))
    avg_txn = _safe_sum(tx, "Net Sales") / max(len(tx), 1)
    k3.metric("Avg Transaction", f"{avg_txn:.2f}")

    # 滚动平均
    daily = tx.copy()
    daily["date"] = daily["Datetime"].dt.floor("D")
    daily_sum = daily.groupby("date", as_index=False)["Net Sales"].sum().sort_values("date")
    daily_sum["3M_Rolling"] = daily_sum["Net Sales"].rolling(90, min_periods=1).mean()
    daily_sum["6M_Rolling"] = daily_sum["Net Sales"].rolling(180, min_periods=1).mean()

    k4.metric("3M Avg", f"{daily_sum['3M_Rolling'].iloc[-1]:.2f}" if not daily_sum.empty else "N/A")
    k5.metric("6M Avg", f"{daily_sum['6M_Rolling'].iloc[-1]:.2f}" if not daily_sum.empty else "N/A")

    # 总库存价值
    total_inv = _safe_sum(inv, "Current Quantity Vie Market & Bar") if inv is not None else 0
    k6.metric("Inventory Value", f"{total_inv:.2f}")

    # 利润（用 Gross Sales - Net Sales 近似）
    profit = _safe_sum(tx, "Gross Sales") - _safe_sum(tx, "Net Sales")
    k7.metric("Profit (Amount)", f"{profit:.2f}")

    k8.metric("Items Sold", f"{_safe_sum(tx, 'Qty'):.0f}")

    # ===== 分类多选折线图 =====
    st.subheader("📈 Daily Net Sales by Location/Category")

    # 🔹 把模拟数据按钮放在图表标题下
    # 🔹 模拟数据控制：下拉 + 生成 + 清空
    st.markdown("**🧪 Data Simulation**")

    option = st.selectbox(
        "Select period for synthetic data",
        ["1M", "3M", "6M", "9M"],
        key="sim_option"
    )

    col1, col2 = st.columns(2)
    if col1.button("Generate Data"):
        months = int(option.replace("M", ""))
        st.session_state["simulated_tx"] = simulate_transactions(tx, months=months)

    if col2.button("❌ Clear Data"):
        if "simulated_tx" in st.session_state:
            del st.session_state["simulated_tx"]

    # 使用模拟数据或真实数据
    if "simulated_tx" in st.session_state:
        tx = st.session_state["simulated_tx"]

    if "location" in tx.columns:
        group_col = "location"
    elif "Category" in tx.columns:
        group_col = "Category"
    else:
        group_col = None

    if group_col:
        cats = sorted(tx[group_col].dropna().astype(str).unique().tolist())
        sel = persisting_multiselect("Choose categories", cats, key="hl_cats")

        df = tx.copy()
        df["date"] = df["Datetime"].dt.floor("D")
        if sel:
            df = df[df[group_col].astype(str).isin(sel)]

        daily_grouped = df.groupby(["date", group_col], as_index=False)["Net Sales"].sum()
        if not daily_grouped.empty:
            fig = px.line(
                daily_grouped,
                x="date",
                y="Net Sales",
                color=group_col,
                title="Daily Net Sales (multi-select)",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected categories.")
    else:
        st.info("No location/category column available.")

    # ===== 第二部分：趋势表格 =====
    st.subheader("📊 Sales Trend Table")
    trend = tx.copy()
    trend["date"] = trend["Datetime"].dt.floor("D")
    trend_stats = trend.groupby("date").agg(
        net_sales=("Net Sales", "sum"),
        transactions=("Datetime", "count"),
        avg_txn=("Net Sales", "mean"),
        profit=("Gross Sales", "sum")
    ).reset_index()

    # 🔹 添加 3M / 6M 滚动平均
    trend_stats["3M_Rolling"] = trend_stats["net_sales"].rolling(90, min_periods=1).mean()
    trend_stats["6M_Rolling"] = trend_stats["net_sales"].rolling(180, min_periods=1).mean()

    if not trend_stats.empty:
        st.dataframe(trend_stats, use_container_width=True)
        st.plotly_chart(
            px.line(
                trend_stats,
                x="date",
                y=["net_sales", "3M_Rolling", "6M_Rolling"],
                title="Sales Trend Over Time (with rolling averages)"
            ),
            use_container_width=True
        )
    else:
        st.info("No trend data available.")
