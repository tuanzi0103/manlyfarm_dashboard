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
            st.plotly_chart(
                px.bar(
                    df_show.head(15),
                    x=item_col,
                    y="restock_needed",
                    title="Items Needing Restock (units)",
                    labels={"restock_needed": "Units to Restock"}
                ),
                width="stretch"
            )
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
            st.plotly_chart(
                px.bar(
                    df_clear.head(15),
                    x=item_col,
                    y="current_qty",
                    title="Items Needing Clearance (units)",
                    labels={"current_qty": "Stock Quantity (units)"}
                ),
                width="stretch"
            )
            cols = [item_col] + [c for c in df_clear.columns if c not in ["row_hash", item_col]]
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
        low["alert_threshold"] = pd.to_numeric(low[threshold_col], errors="coerce")
        low["alert_threshold"] = low["alert_threshold"].fillna(default_threshold)
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
            first_col = item_col
            cols = [first_col] + [c for c in filtered.columns if c not in ["row_hash", first_col]]
            st.dataframe(filtered[cols], use_container_width=True)

            st.plotly_chart(
                px.bar(
                    filtered.head(20),
                    x=first_col,
                    y="current_qty",
                    title="Low Stock Items",
                    labels={"current_qty": "Stock Quantity (units)"}
                ),
                width="stretch"
            )
        else:
            st.info("No matching low-stock items found.")
    else:
        st.success("No low-stock items.")

    # ---- 3) Future Consumption Forecast ----
    st.subheader("3) Forecasted Consumption for the Next Month")
    st.caption(
        "Method: Uses recent **7-day daily average × 30** for simple forecasting; "
        "search below to view history and forecast curve for a specific item (units/day)."
    )

    # 模拟数据按钮
    cols = st.columns([1, 1, 1, 1, 2])
    if cols[0].button("Generate 1M Test Data", key="gen1m"):
        st.session_state["test_tx"] = simulate_consumption(inventory, months=1)
    if cols[1].button("Generate 3M Test Data", key="gen3m"):
        st.session_state["test_tx"] = simulate_consumption(inventory, months=3)
    if cols[2].button("Generate 6M Test Data", key="gen6m"):
        st.session_state["test_tx"] = simulate_consumption(inventory, months=6)
    if cols[3].button("Generate 9M Test Data", key="gen9m"):
        st.session_state["test_tx"] = simulate_consumption(inventory, months=9)
    if cols[4].button("Clear Test Data", key="clear_test"):
        st.session_state.pop("test_tx", None)

    tx_used = st.session_state.get("test_tx", tx)

    # 🔹 自动检测 item_col，避免 KeyError
    if "Item" in tx_used.columns:
        item_col = "Item"
    elif "Item Name" in tx_used.columns:
        item_col = "Item Name"
    elif "SKU" in tx_used.columns:
        item_col = "SKU"
    else:
        item_col = tx_used.columns[0]  # 兜底

    options = sorted(tx_used[item_col].astype(str).unique())
    selected_items = st.multiselect("Search by Item Name or SKU", options, key="sku_multi")

    if "test_tx" in st.session_state:
        # 使用模拟处理逻辑
        ds, fc = simulate_consumption_timeseries(inventory, months=3, items=selected_items if selected_items else None)
    else:
        # 原逻辑
        if selected_items:
            all_ds, all_fc = [], []
            for q in selected_items:
                ds, fc = sku_consumption_timeseries(tx_used, q)
                if not ds.empty:
                    ds["Item"] = q
                    all_ds.append(ds)
                if not fc.empty:
                    fc["Item"] = q
                    all_fc.append(fc)
            ds = pd.concat(all_ds, ignore_index=True) if all_ds else pd.DataFrame()
            fc = pd.concat(all_fc, ignore_index=True) if all_fc else pd.DataFrame()
        else:
            ds, fc = None, None

    # Top consumers
    top_consume = forecast_top_consumers(tx_used, topn=15)
    if not top_consume.empty:
        top_consume = top_consume[top_consume["forecast_30d"].notna() & (top_consume["forecast_30d"] > 0)]
        if not top_consume.empty:
            st.plotly_chart(
                px.bar(
                    top_consume,
                    x="Item",
                    y="forecast_30d",
                    color="increasing",
                    title="Top Predicted Consumption in Next 30 Days (units)",
                    labels={"forecast_30d": "Total Forecast Consumption (units)"}
                ),
                width="stretch"
            )

    # 预测曲线
    if fc is not None and not fc.empty:
        st.plotly_chart(
            px.line(
                fc, x="date", y="forecast_qty", color="Item" if "Item" in fc.columns else None,
                title="30-Day Consumption Forecast (units/day)",
                labels={"forecast_qty": "Forecasted Daily Consumption (units/day)"}
            ),
            width="stretch"
        )
