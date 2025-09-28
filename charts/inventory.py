import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional

from services.analytics import (
    forecast_top_consumers,
    sku_consumption_timeseries,
)


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

    # Items needing restock = 当前数量为负数
    need_restock = inv[inv[qty_col] < 0].copy()
    if not need_restock.empty:
        df_show = need_restock.copy()
        df_show["restock_needed"] = df_show[qty_col].abs()
        st.plotly_chart(
            px.bar(
                df_show.head(15),
                x="Item" if "Item" in df_show.columns else "Item Name",
                y="restock_needed",
                title="Items Needing Restock (units)",
                labels={"restock_needed": "Units to Restock"}
            ),
            use_container_width=True
        )
        st.dataframe(
            df_show[[c for c in df_show.columns if c in ["Item", "Item Name", qty_col, "restock_needed"]]],
            use_container_width=True
        )
    else:
        st.success("No items need restocking.")

    # Items needing clearance = 数量非常大的（这里保留原有逻辑，阈值=50）
    clear_threshold = 50
    need_clear = inv[pd.to_numeric(inv[qty_col], errors="coerce").fillna(0) > clear_threshold].copy()
    if not need_clear.empty:
        df_clear = need_clear.copy()
        df_clear["current_qty"] = pd.to_numeric(df_clear[qty_col], errors="coerce").fillna(0).abs()
        st.plotly_chart(
            px.bar(
                df_clear.head(15),
                x="Item" if "Item" in df_clear.columns else "Item Name",
                y="current_qty",
                title="Items Needing Clearance (units)",
                labels={"current_qty": "Stock Quantity (units)"}
            ),
            use_container_width=True
        )
        st.dataframe(df_clear, use_container_width=True)
    else:
        st.info("No items need clearance.")

    # ---- 2) Low Stock Alerts ----
    st.subheader("2) Low Stock Alerts")

    # 读取 threshold 列
    threshold_col = None
    for c in inv.columns:
        if "stock alert count" in str(c).lower():
            threshold_col = c
            break

    # 用户可设置默认阈值（当列值为空时使用）
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
        # 如果表里没有这个列，统一使用 default_threshold
        low["alert_threshold"] = default_threshold

    # 低库存判定
    low = low[low["current_qty"] <= low["alert_threshold"]]

    if not low.empty:
        search_q = st.text_input("Search by Item Name or SKU", key="lowstock_q")
        filtered = low.copy()

        if search_q:
            mask = False
            for col in ["Item", "Item Name", "SKU", "sku"]:
                if col in filtered.columns:
                    mask = mask | filtered[col].astype(str).str.contains(search_q, case=False, na=False)
            filtered = filtered[mask]

        if not filtered.empty:
            st.dataframe(filtered, use_container_width=True)

            # 绘图
            x_col = None
            for c in ["Item", "Item Name", "SKU", "sku"]:
                if c in filtered.columns:
                    x_col = c
                    break

            if x_col:
                st.plotly_chart(
                    px.bar(
                        filtered.head(20),
                        x=x_col,
                        y="current_qty",
                        title="Low Stock Items",
                        labels={"current_qty": "Stock Quantity (units)"}
                    ),
                    use_container_width=True
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

    query = st.text_input("Search by Item Name", key="sku_q")
    ds, fc = sku_consumption_timeseries(tx, query) if query else (None, None)

    top_consume = forecast_top_consumers(tx, topn=15)
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
                use_container_width=True
            )
        else:
            st.info("No valid forecast data available.")
    else:
        st.info("Not enough data to generate forecasts.")

    if ds is not None and fc is not None and not ds.empty:
        st.plotly_chart(
            px.line(
                ds, x="date", y="qty",
                title="Historical Daily Consumption (units/day)",
                labels={"qty": "Daily Consumption (units/day)"}
            ),
            use_container_width=True
        )
        st.plotly_chart(
            px.line(
                fc, x="date", y="forecast_qty",
                title="30-Day Consumption Forecast (units/day)",
                labels={"forecast_qty": "Forecasted Daily Consumption (units/day)"}
            ),
            use_container_width=True
        )
