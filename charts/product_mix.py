import streamlit as st
import plotly.express as px
from services.analytics import (
    restock_and_overstock_tables,
    inventory_low_stock_alert,
    forecast_top_consumers,
    sku_consumption_timeseries,
)


def show_product_mix(tx, inventory):
    st.header("📦 Product Mix & Inventory Optimization")

    if tx.empty:
        st.info("No transaction data available")
        return

    # Thresholds
    restock_threshold = st.number_input("Restock Threshold (≤ triggers restock)", value=3.0, step=1.0)
    clear_threshold = st.number_input("Clearance Threshold (> triggers clearance)", value=50.0, step=1.0)

    # 1) Restock / Clearance
    st.subheader("1) Inventory Diagnosis: Restock / Clearance Needed")
    need_restock, need_clear = restock_and_overstock_tables(
        inventory,
        restock_threshold=restock_threshold,
        clear_threshold=clear_threshold,
        topn=30
    )

    if not need_restock.empty:
        y_min, y_max = need_restock["current_qty"].min(), need_restock["current_qty"].max()
        st.plotly_chart(
            px.bar(
                need_restock.head(15),
                x="Item" if "Item" in need_restock.columns else "Item Name",
                y="current_qty",
                title="Items Needing Restock (units)",
                labels={"current_qty": "Stock Quantity (units)"}
            ).update_yaxes(title="Stock Quantity (units)", range=[y_min, y_max]),
            use_container_width=True
        )
        st.dataframe(need_restock, use_container_width=True)
    else:
        st.success("No items need restocking.")

    if not need_clear.empty:
        need_clear = need_clear[need_clear["current_qty"] > clear_threshold]
        if not need_clear.empty:
            y_min, y_max = need_clear["current_qty"].min(), need_clear["current_qty"].max()
            st.plotly_chart(
                px.bar(
                    need_clear.head(15),
                    x="Item" if "Item" in need_clear.columns else "Item Name",
                    y="current_qty",
                    title="Items Needing Clearance (units)",
                    labels={"current_qty": "Stock Quantity (units)"}
                ).update_yaxes(title="Stock Quantity (units)", range=[y_min, y_max]),
                use_container_width=True
            )
            st.dataframe(need_clear, use_container_width=True)
        else:
            st.info("No obvious clearance items found.")
    else:
        st.info("No items need clearance.")

    # 2) Low Stock Alerts
    st.subheader("2) Low Stock Alerts")
    low = inventory_low_stock_alert(inventory, threshold=restock_threshold)
    if not low.empty:
        search_q = st.text_input("Search by Item Name or SKU", key="lowstock_q")

        if search_q:
            # 有搜索时 → 只显示过滤后的表格
            filtered = low.copy()
            mask = False
            for col in ["Item", "Item Name", "SKU", "sku"]:
                if col in filtered.columns:
                    mask = mask | filtered[col].astype(str).str.contains(search_q, case=False, na=False)
            filtered = filtered[mask]

            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True)
            else:
                st.info("No matching low-stock items found.")
        else:
            # 默认状态 → 显示完整表格 + 柱形图
            st.dataframe(low, use_container_width=True)

            # 动态选择 X、Y 列
            x_col = None
            for c in ["Item", "Item Name", "SKU", "sku"]:
                if c in low.columns:
                    x_col = c
                    break

            y_col = None
            for c in ["current_qty", "stock_qty", "qty"]:
                if c in low.columns:
                    y_col = c
                    break

            if x_col and y_col:
                st.plotly_chart(
                    px.bar(
                        low.head(20),
                        x=x_col,
                        y=y_col,
                        title="Low Stock Items",
                        labels={y_col: "Stock Quantity (units)"}
                    ),
                    use_container_width=True
                )
    else:
        st.success("No low-stock items.")

    # 3) Future Consumption Forecast
    st.subheader("3) Forecasted Consumption for the Next Month")
    st.caption(
        "Method: Uses recent **7-day daily average × 30** for simple forecasting; "
        "search below to view history and forecast curve for a specific item (units/day)."
    )

    query = st.text_input("Search by Item Name", key="sku_q")
    ds, fc = sku_consumption_timeseries(tx, query) if query else (None, None)

    top_consume = forecast_top_consumers(tx, topn=15)
    if not top_consume.empty:
        # Filter out NaN or <=0 values
        top_consume = top_consume[top_consume["forecast_30d"].notna() & (top_consume["forecast_30d"] > 0)]

        if not top_consume.empty:
            y_min, y_max = top_consume["forecast_30d"].min(), top_consume["forecast_30d"].max()
            st.plotly_chart(
                px.bar(
                    top_consume,
                    x="Item",
                    y="forecast_30d",
                    color="increasing",
                    title="Top Predicted Consumption in Next 30 Days (units)",
                    labels={"forecast_30d": "Total Forecast Consumption (units)"}
                ).update_yaxes(title="Total Forecast Consumption (units)", range=[y_min, y_max]),
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
