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

    # ---- 💰 Inventory Valuation Analysis ----
    st.subheader("💰 Inventory Valuation Analysis")

    time_range = st.multiselect(
        "Choose Time Range", ["WTD", "MTD", "YTD"], key="inv_timerange"
    )
    all_items = sorted(inv["Item Name"].fillna("Unknown").unique().tolist()) if "Item Name" in inv.columns else []
    bar_cats = ["Café Drinks", "Smoothie bar", "Soups", "Sweet Treats", "Wrap & Salads"]

    categories = st.multiselect("Choose Categories / Items", all_items + ["bar", "retail"], key="inv_category")

    if time_range and categories:
        df = inv.copy()

        # 必要列
        needed_cols = [
            "Item Name", "Item Variation Name", "GTIN", "SKU",
            "Current Quantity Vie Market & Bar", "Tax - GST (10%)", "Price", "Default Unit Cost", "Categories"
        ]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = 0

        # ✅ Quantity 取绝对值
        df["Quantity"] = pd.to_numeric(df["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0).abs()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce").fillna(0)

        # === 计算 Total Retail Value / Inventory Value / Profit ===
        def calc_retail(row):
            O, AA, tax = row["Price"], row["Quantity"], str(row["Tax - GST (10%)"]).strip().upper()
            return (O / 11 * 10) * AA if tax == "Y" else O * AA

        df["Total Retail Value"] = df.apply(calc_retail, axis=1)
        df["Total Inventory Value"] = df["UnitCost"] * df["Quantity"]
        df["Profit"] = df["Total Retail Value"] - df["Total Inventory Value"]

        # ========== 构建结果 ==========
        results = []

        # 1) 单一小类 → 按 Item Name 精确过滤
        selected_items = [c for c in categories if c not in ["bar", "retail"]]
        if selected_items:
            small_df = df[df["Item Name"].isin(selected_items)].copy()
            if not small_df.empty:
                small_df = small_df[
                    ["Item Name", "Item Variation Name", "GTIN", "SKU", "Quantity",
                     "Total Inventory Value", "Total Retail Value", "Profit"]
                ]
                small_df["Profit Margin"] = (small_df["Profit"] / small_df["Total Retail Value"] * 100).fillna(0)
                results.append(small_df)

        # 2) bar 聚合 → 一行
        if "bar" in categories:
            bar_df = df[df["Categories"].isin(bar_cats)]
            if not bar_df.empty:
                agg = {
                    "Item Name": "BAR (All)",
                    "Item Variation Name": "",
                    "GTIN": "",
                    "SKU": "",
                    "Quantity": bar_df["Quantity"].sum(),
                    "Total Inventory Value": bar_df["Total Inventory Value"].sum(),
                    "Total Retail Value": bar_df["Total Retail Value"].sum(),
                    "Profit": bar_df["Profit"].sum(),
                }
                agg["Profit Margin"] = (
                    agg["Profit"] / agg["Total Retail Value"] * 100
                    if agg["Total Retail Value"] > 0 else 0
                )
                agg["Velocity"] = (
                    tx["Net Sales"].sum() / agg["Total Retail Value"]
                    if agg["Total Retail Value"] > 0 else 0
                )
                results.append(pd.DataFrame([agg]))

        # 3) retail 聚合 → 一行
        if "retail" in categories:
            retail_df = df[~df["Categories"].isin(bar_cats)]
            if not retail_df.empty:
                agg = {
                    "Item Name": "RETAIL (All)",
                    "Item Variation Name": "",
                    "GTIN": "",
                    "SKU": "",
                    "Quantity": retail_df["Quantity"].sum(),
                    "Total Inventory Value": retail_df["Total Inventory Value"].sum(),
                    "Total Retail Value": retail_df["Total Retail Value"].sum(),
                    "Profit": retail_df["Profit"].sum(),
                }
                agg["Profit Margin"] = (
                    agg["Profit"] / agg["Total Retail Value"] * 100
                    if agg["Total Retail Value"] > 0 else 0
                )
                agg["Velocity"] = (
                    tx["Net Sales"].sum() / agg["Total Retail Value"]
                    if agg["Total Retail Value"] > 0 else 0
                )
                results.append(pd.DataFrame([agg]))

        if results:
            df_show = pd.concat(results, ignore_index=True)
            # 格式化 Profit Margin 为百分比显示
            if "Profit Margin" in df_show.columns:
                df_show["Profit Margin"] = df_show["Profit Margin"].map(lambda x: f"{x:.2f}%")
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("No data for selected categories.")

    else:
        st.info("Please select both Time Range and Category to view valuation table.")


    # ---- 1) Inventory Diagnosis: Restock / Clearance ----
    st.subheader("1) Inventory Diagnosis: Restock / Clearance Needed")
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

    # Items needing restock
    need_restock = inv[inv[qty_col] < 0].copy()
    if not need_restock.empty:
        options = sorted(need_restock["option_key"].unique())
        selected_items = st.multiselect("Search/Filter Items (Restock)", options, key="restock_filter")
        df_show = need_restock.copy()
        df_show["restock_needed"] = df_show[qty_col].abs()
        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_show = df_show[df_show["SKU"].astype(str).isin(selected_skus)]
            else:
                df_show = df_show[df_show["display_name"].isin(selected_items)]
        if not df_show.empty:
            st.plotly_chart(px.bar(df_show.head(15), x="display_name", y="restock_needed",
                                   title="Items Needing Restock (units)",
                                   labels={"restock_needed": "Units to Restock"}), use_container_width=True)
            st.dataframe(df_show[[c for c in df_show.columns if c in ["display_name", qty_col, "restock_needed"]]],
                         use_container_width=True)
        else:
            st.info("No matching items to restock.")
    else:
        st.success("No items need restocking.")

    # Items needing clearance
    clear_threshold = 50
    need_clear = inv[pd.to_numeric(inv[qty_col], errors="coerce").fillna(0) > clear_threshold].copy()
    if not need_clear.empty:
        options = sorted(need_clear["option_key"].unique())
        selected_items = st.multiselect("Search/Filter Items (Clearance)", options, key="clear_filter")
        df_clear = need_clear.copy()
        df_clear["current_qty"] = pd.to_numeric(df_clear[qty_col], errors="coerce").fillna(0).abs()
        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_clear = df_clear[df_clear["SKU"].astype(str).isin(selected_skus)]
            else:
                df_clear = df_clear[df_clear["display_name"].isin(selected_items)]
        if not df_clear.empty:
            st.plotly_chart(px.bar(df_clear.head(15), x="display_name", y="current_qty",
                                   title="Items Needing Clearance (units)",
                                   labels={"current_qty": "Stock Quantity (units)"}), use_container_width=True)
            cols = ["display_name"] + [c for c in df_clear.columns if c not in ["row_hash", "display_name"]]
            st.dataframe(df_clear[cols], use_container_width=True)
        else:
            st.info("No matching items need clearance.")
    else:
        st.info("No items need clearance.")

    # ---- 2) Low Stock Alerts ----
    st.subheader("2) Low Stock Alerts")
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
        options = sorted(low["option_key"].unique())
        selected_items = st.multiselect("Search/Filter Items (Low Stock)", options, key="lowstock_filter")
        filtered = low.copy()
        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                filtered = filtered[filtered["SKU"].astype(str).isin(selected_skus)]
            else:
                filtered = filtered[filtered["display_name"].isin(selected_items)]
        if not filtered.empty:
            st.dataframe(filtered[[c for c in filtered.columns if c not in ["row_hash"]]], use_container_width=True)
            st.plotly_chart(px.bar(filtered.head(20), x="display_name", y="current_qty",
                                   title="Low Stock Items", labels={"current_qty": "Stock Quantity (units)"}),
                            use_container_width=True)
        else:
            st.info("No matching low-stock items found.")
    else:
        st.success("No low-stock items.")

    # ⚠️ 已删除 3) Forecasted Consumption 和旧的 4)
