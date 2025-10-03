import streamlit as st
import pandas as pd
import plotly.express as px
from services.db import get_db

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

def persisting_multiselect(label, options, key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default or []
    return st.multiselect(label, options, default=st.session_state[key], key=key)

# === 用“按收据(Transaction ID)”聚合，保证口径与图1一致 ===
@st.cache_data
def get_high_level_data():
    db = get_db()

    daily_sql = """
        WITH tx AS (
        SELECT 
            date(Datetime) AS date,
            [Transaction ID] AS txn_id,
            SUM([Net Sales])  AS tx_net,
            SUM([Gross Sales]) AS tx_gross,
            SUM(Qty)          AS tx_qty
        FROM transactions
        GROUP BY date, [Transaction ID]
    )
    SELECT
        date,
        SUM(CASE WHEN tx_net > 0 THEN tx_net ELSE 0 END)                         AS net_sales,
        SUM(CASE WHEN tx_net > 0 THEN 1      ELSE 0 END)                         AS transactions,
        (CASE WHEN SUM(CASE WHEN tx_net > 0 THEN 1 ELSE 0 END) > 0
              THEN SUM(CASE WHEN tx_net > 0 THEN tx_net ELSE 0 END) * 1.0
                   / SUM(CASE WHEN tx_net > 0 THEN 1 ELSE 0 END)
         END)                                                                    AS avg_txn,
        SUM(tx_gross)                                                            AS gross,
        SUM(tx_qty)                                                              AS qty
    FROM tx
    GROUP BY date
    ORDER BY date;
    """

    category_sql = """
    WITH line AS (
        SELECT 
            date(Datetime) AS date,
            Category,
            [Transaction ID] AS txn_id,
            [Net Sales]      AS net_sales,
            [Gross Sales]    AS gross,
            Qty              AS qty
        FROM transactions
    ),
    cat_tx AS (
        SELECT
            date,
            Category,
            txn_id,
            SUM(net_sales) AS cat_tx_net,
            SUM(gross)     AS cat_tx_gross,
            SUM(qty)       AS cat_tx_qty
        FROM line
        GROUP BY date, Category, txn_id
    )
    SELECT
        date,
        Category,
        SUM(CASE WHEN cat_tx_net > 0 THEN cat_tx_net ELSE 0 END)  AS net_sales,
        SUM(CASE WHEN cat_tx_net > 0 THEN 1          ELSE 0 END)  AS transactions,
        AVG(CASE WHEN cat_tx_net > 0 THEN cat_tx_net END)         AS avg_txn,
        SUM(cat_tx_gross)                                         AS gross,
        SUM(cat_tx_qty)                                           AS qty
    FROM cat_tx
    GROUP BY date, Category
    ORDER BY date;
    """

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])

    return daily, category


@st.cache_data
def _prepare_inventory_grouped(inv: pd.DataFrame):
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


def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame):
    st.header("📊 High Level Report")

    daily, category_tx = get_high_level_data()
    inv_grouped, inv_latest_date = _prepare_inventory_grouped(inv)

    if daily.empty:
        st.warning("No transaction data available. Please upload data first.")
        return

    today = pd.Timestamp.today().normalize()
    latest_date_tx = daily["date"].max()
    df_latest_tx = daily[daily["date"] == latest_date_tx]

    # === KPI（交易，口径按小票） ===
    kpis_main = {
        "Daily Net Sales": df_latest_tx["net_sales"].sum(),
        "Daily Transactions": df_latest_tx["transactions"].sum(),
        "Avg Transaction": df_latest_tx["avg_txn"].mean(),
        "3M Avg": daily["net_sales"].rolling(90, min_periods=1).mean().iloc[-1],
        "6M Avg": daily["net_sales"].rolling(180, min_periods=1).mean().iloc[-1],
        "Items Sold": df_latest_tx["qty"].sum(),
    }

    # === KPI（库存派生，catalogue-only） ===
    inv_value_latest = 0.0
    profit_latest = 0.0
    if inv_grouped is not None and not inv_grouped.empty and inv_latest_date is not None:
        sub = inv_grouped[inv_grouped["date"] == inv_latest_date]
        inv_value_latest = float(pd.to_numeric(sub["Inventory Value"], errors="coerce").sum())
        profit_latest = float(pd.to_numeric(sub["Profit"], errors="coerce").sum())

    st.markdown(f"### 📅 Latest TX date: {pd.to_datetime(latest_date_tx).strftime('%Y-%m-%d')}")
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
                display = "-" if pd.isna(val) else f"{val:,.2f}"
                with col:
                    st.markdown(f"<div style='font-size:28px; font-weight:600'>{display}</div>", unsafe_allow_html=True)
                    st.caption(label)
                    if label in captions:
                        st.caption(captions[label])

    st.markdown("---")

    # === 交互选择 ===
    st.subheader("🔍 Select Parameters")
    time_range = persisting_multiselect("Choose time range", ["Custom dates", "WTD", "MTD", "YTD"], key="hl_time")
    data_options = [
        "Daily Net Sales","Daily Transactions","Avg Transaction","3M Avg","6M Avg",
        "Inventory Value","Profit (Amount)","Items Sold"
    ]
    data_sel = persisting_multiselect("Choose data type", data_options, key="hl_data")

    bar_cats = {"Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wraps & Salads"}

    if category_tx is None or category_tx.empty:
        st.info("No category breakdown available.")
        return

    all_cats_tx = sorted(category_tx["Category"].fillna("Unknown").unique().tolist())
    all_cats_extended = sorted(set(all_cats_tx + ["bar", "retail"]))
    cats_sel = persisting_multiselect("Choose categories", all_cats_extended, key="hl_cats")

    if time_range and data_sel and cats_sel:
        grouped_tx = category_tx.copy()

        if "WTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=7)]
        if "MTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=30)]
        if "YTD" in time_range:
            grouped_tx = grouped_tx[grouped_tx["date"] >= today - pd.Timedelta(days=365)]
        if "Custom dates" in time_range:
            t1 = st.date_input("From")
            t2 = st.date_input("To")
            if t1 and t2:
                grouped_tx = grouped_tx[(grouped_tx["date"] >= pd.to_datetime(t1)) & (grouped_tx["date"] <= pd.to_datetime(t2))]

        grouped_inv = inv_grouped.copy()
        if not grouped_inv.empty:
            if "WTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=7)]
            if "MTD" in time_range:
                grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=30)]
            # 原来：
            # if "YTD" in time_range:
            #     grouped_inv = grouped_inv[grouped_inv["date"] >= today - pd.Timedelta(days=365)]

            # 改为（过去365天到今天，含上限）：
            if "YTD" in time_range:
                ytd_start = today - pd.Timedelta(days=365)
                ytd_end = today
                grouped_inv = grouped_inv[(grouped_inv["date"] >= ytd_start) & (grouped_inv["date"] <= ytd_end)]

        small_cats = [c for c in cats_sel if c not in ("bar", "retail")]
        parts_tx = []

        if small_cats:
            parts_tx.append(grouped_tx[grouped_tx["Category"].isin(small_cats)])

        if "bar" in cats_sel:
            bar_df = grouped_tx[grouped_tx["Category"].isin(list(bar_cats))]
            if not bar_df.empty:
                agg = (bar_df.groupby("date", as_index=False)
                       .agg(net_sales=("net_sales","sum"),
                            transactions=("transactions","sum"),
                            gross=("gross","sum"),
                            qty=("qty","sum")))
                agg["avg_txn"] = (agg["net_sales"] / agg["transactions"]).replace([pd.NA, float("inf")], 0)
                agg["Category"] = "bar"
                parts_tx.append(agg)

        if "retail" in cats_sel:
            retail_df = grouped_tx[~grouped_tx["Category"].isin(list(bar_cats))]
            if not retail_df.empty:
                agg = (retail_df.groupby("date", as_index=False)
                       .agg(net_sales=("net_sales","sum"),
                            transactions=("transactions","sum"),
                            gross=("gross","sum"),
                            qty=("qty","sum")))
                agg["avg_txn"] = (agg["net_sales"] / agg["transactions"]).replace([pd.NA, float("inf")], 0)
                agg["Category"] = "retail"
                parts_tx.append(agg)

        grouped_tx = pd.concat(parts_tx, ignore_index=True) if parts_tx else grouped_tx.iloc[0:0]

        parts_inv = []
        if not grouped_inv.empty:
            if small_cats:
                parts_inv.append(grouped_inv[grouped_inv["Category"].isin(small_cats)])

            if "bar" in cats_sel:
                bar_inv = grouped_inv[grouped_inv["Category"].isin(list(bar_cats))]
                if not bar_inv.empty:
                    agg = (bar_inv.groupby("date", as_index=False)
                           .agg(**{"Inventory Value":("Inventory Value", "sum"),
                                   "Profit":("Profit", "sum")}))
                    agg["Category"] = "bar"
                    parts_inv.append(agg)

            if "retail" in cats_sel:
                retail_inv = grouped_inv[~grouped_inv["Category"].isin(list(bar_cats))]
                if not retail_inv.empty:
                    agg = (retail_inv.groupby("date", as_index=False)
                           .agg(**{"Inventory Value":("Inventory Value", "sum"),
                                   "Profit":("Profit", "sum")}))
                    agg["Category"] = "retail"
                    parts_inv.append(agg)

        grouped_inv = pd.concat(parts_inv, ignore_index=True) if parts_inv else grouped_inv.iloc[0:0]

        mapping_tx = {
            "Daily Net Sales": ("net_sales", "Daily Net Sales"),
            "Daily Transactions": ("transactions", "Daily Transactions"),
            "Avg Transaction": ("avg_txn", "Avg Transaction"),
            "3M Avg": ("net_sales", "3M Avg (Rolling 90d)"),
            "6M Avg": ("net_sales", "6M Avg (Rolling 180d)"),
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
                        plot_df["rolling"] = plot_df.groupby("Category")[y].transform(lambda x: x.rolling(90, min_periods=1).mean())
                    else:
                        plot_df["rolling"] = plot_df.groupby("Category")[y].transform(lambda x: x.rolling(180, min_periods=1).mean())
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
            tables.append(grouped_tx[cols_tx].assign(date=grouped_tx["date"].dt.strftime("%Y-%m-%d")))

        if not grouped_inv.empty:
            cols_inv = ["date", "Category"]
            for sel in data_sel:
                if sel in mapping_inv:
                    cols_inv.append(mapping_inv[sel][0])
            tables.append(grouped_inv[cols_inv].assign(date=grouped_inv["date"].dt.strftime("%Y-%m-%d")))

        if tables:
            out = pd.concat(tables, ignore_index=True)
            st.dataframe(out, use_container_width=True)
        else:
            st.info("No data for the selected filters.")
    else:
        st.info("Please select time range, data, and category to generate the chart.")
