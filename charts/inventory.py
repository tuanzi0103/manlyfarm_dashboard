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


def show_inventory(tx, inventory: pd.DataFrame):
    st.header("📦 Product Mix & Inventory Optimization")

    if tx.empty:
        st.info("No transaction data available")
        return

    if inventory is None or inventory.empty:
        st.info("No inventory data available")
        return

    inv = inventory.copy()

    # 找出数量列
    qty_col = detect_store_current_qty_col(inv)
    if qty_col is None:
        st.warning("No valid quantity column found in inventory file.")
        return

    # ---- 1) Inventory Diagnosis: Restock / Clearance ----
    st.subheader("1) Inventory Diagnosis: Restock / Clearance Needed")
    # ...（原逻辑保持不变）
    # Items needing restock
    need_restock = inv[inv[qty_col] < 0].copy()
    if not need_restock.empty:
        item_col = "Item" if "Item" in need_restock.columns else "Item Name"
        options = sorted(need_restock[item_col].astype(str).unique())
        selected_items = st.multiselect("Search/Filter Items (Restock)", options, key="restock_filter")
        df_show = need_restock.copy()
        df_show["restock_needed"] = df_show[qty_col].abs()
        if selected_items:
            df_show = df_show[df_show[item_col].isin(selected_items)]
        if not df_show.empty:
            st.plotly_chart(px.bar(df_show.head(15), x=item_col, y="restock_needed",
                                   title="Items Needing Restock (units)",
                                   labels={"restock_needed": "Units to Restock"}), use_container_width=True)
            st.dataframe(df_show[[c for c in df_show.columns if c in [item_col, qty_col, "restock_needed"]]],
                         use_container_width=True)
        else:
            st.info("No matching items to restock.")
    else:
        st.success("No items need restocking.")

    # Items needing clearance
    clear_threshold = 50
    need_clear = inv[pd.to_numeric(inv[qty_col], errors="coerce").fillna(0) > clear_threshold].copy()
    if not need_clear.empty:
        item_col = "Item" if "Item" in need_clear.columns else "Item Name"
        options = sorted(need_clear[item_col].astype(str).unique())
        selected_items = st.multiselect("Search/Filter Items (Clearance)", options, key="clear_filter")
        df_clear = need_clear.copy()
        df_clear["current_qty"] = pd.to_numeric(df_clear[qty_col], errors="coerce").fillna(0).abs()
        if selected_items:
            df_clear = df_clear[df_clear[item_col].isin(selected_items)]
        if not df_clear.empty:
            st.plotly_chart(px.bar(df_clear.head(15), x=item_col, y="current_qty",
                                   title="Items Needing Clearance (units)",
                                   labels={"current_qty": "Stock Quantity (units)"}), use_container_width=True)
            cols = [item_col] + [c for c in df_clear.columns if c not in ["row_hash", item_col]]
            st.dataframe(df_clear[cols], use_container_width=True)
        else:
            st.info("No matching items need clearance.")
    else:
        st.info("No items need clearance.")

    # ---- 2) Low Stock Alerts ----
    st.subheader("2) Low Stock Alerts")
    # ...（原逻辑保持不变）
    threshold_col = None
    for c in inv.columns:
        if "stock alert count" in str(c).lower():
            threshold_col = c
            break

    default_threshold = st.number_input(
        "Default Low Stock Threshold (applies when 'Stock Alert Count' is empty)",
        min_value=1, value=2, step=1
    )

    low = inv.copy()
    low["current_qty"] = pd.to_numeric(low[qty_col], errors="coerce").fillna(0).abs()
    if threshold_col:
        low["alert_threshold"] = pd.to_numeric(low[threshold_col], errors="coerce").fillna(default_threshold)
    else:
        low["alert_threshold"] = default_threshold

    low = low[low["current_qty"] <= low["alert_threshold"]]
    if not low.empty:
        item_col = "Item" if "Item" in low.columns else "Item Name"
        options = sorted(low[item_col].astype(str).unique())
        selected_items = st.multiselect("Search/Filter Items (Low Stock)", options, key="lowstock_filter")
        filtered = low.copy()
        if selected_items:
            filtered = filtered[filtered[item_col].isin(selected_items)]
        if not filtered.empty:
            st.dataframe(filtered[[c for c in filtered.columns if c not in ["row_hash"]]], use_container_width=True)
            st.plotly_chart(px.bar(filtered.head(20), x=item_col, y="current_qty",
                                   title="Low Stock Items", labels={"current_qty": "Stock Quantity (units)"}),
                            use_container_width=True)
        else:
            st.info("No matching low-stock items found.")
    else:
        st.success("No low-stock items.")

    # ---- 3) Future Consumption Forecast ----
    st.subheader("3) Forecasted Consumption for the Next Month")
    # ...（原逻辑保持不变，省略不改）

    # ---- 4) 新增：Inventory Valuation Analysis ----
    st.subheader("4) 💰 Inventory Valuation Analysis")

    # Time Range
    time_range = st.multiselect("Choose Time Range", ["Custom dates", "WTD", "MTD", "YTD"], key="inv_timerange")
    # Category
    all_cats = sorted(inv["Category"].fillna("Unknown").unique().tolist()) if "Category" in inv.columns else []
    bar_cats = ["Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"]
    categories = st.multiselect("Choose Categories", all_cats + ["bar", "retail"], key="inv_category")

    if time_range and categories:
        df = inv.copy()
        # 分类处理
        if "Category" in df.columns:
            df["cat_group"] = df["Category"].fillna("Unknown")
            df.loc[df["Category"].isin(bar_cats), "cat_group"] = "bar"
            df.loc[~df["Category"].isin(bar_cats), "cat_group"] = "retail"
            df = df[df["cat_group"].isin(categories)]
        # 必要列
        needed_cols = ["Item Name", "Item Variation Name", "GTIN", "SKU", "Quantity", "Tax - GST (10%)", "Price"]
        available_cols = [c for c in needed_cols if c in df.columns]
        df = df[available_cols + ["cat_group"]].copy()
        df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)
        df["Price"] = pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0)

        def calc_retail(row):
            O = row["Price"]
            AA = row["Quantity"]
            tax = str(row.get("Tax - GST (10%)", "N")).strip().upper()
            return (O / 11 * 10) * AA if tax == "Y" else O * AA

        df["Total Retail Value"] = df.apply(calc_retail, axis=1)
        df["Total Inventory Value"] = df["Price"] * df["Quantity"]
        df["Profit"] = df["Total Retail Value"] - df["Total Inventory Value"]
        df["Profit Margin"] = df["Profit"] / df["Total Retail Value"].replace(0, 1)

        # Velocity: 用选定 time_range 的 tx
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        tx_filtered = tx.copy()
        if "WTD" in time_range:
            tx_filtered = tx_filtered[tx_filtered["date"] >= today - pd.Timedelta(days=7)]
        if "MTD" in time_range:
            tx_filtered = tx_filtered[tx_filtered["date"] >= today - pd.Timedelta(days=30)]
        if "YTD" in time_range:
            tx_filtered = tx_filtered[tx_filtered["date"] >= today - pd.Timedelta(days=365)]
        if "Custom dates" in time_range:
            t1 = st.date_input("From")
            t2 = st.date_input("To")
            if t1 and t2:
                tx_filtered = tx_filtered[(tx_filtered["date"] >= pd.Timestamp(t1)) & (tx_filtered["date"] <= pd.Timestamp(t2))]

        monthly_sales = tx_filtered.groupby("Category")["Net Sales"].sum().to_dict() if "Category" in tx_filtered.columns else {}
        total_retail_sum = df["Total Retail Value"].sum()
        df["Velocity"] = df["cat_group"].map(lambda c: monthly_sales.get(c, 0) / total_retail_sum if total_retail_sum > 0 else 0)

        show_cols = ["Item Name", "Item Variation Name", "GTIN", "SKU", "Quantity",
                     "Total Inventory Value", "Total Retail Value", "Profit", "Profit Margin", "Velocity"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)
    else:
        st.info("Please select both Time Range and Category to view valuation table.")
