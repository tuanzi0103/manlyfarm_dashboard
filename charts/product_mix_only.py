import streamlit as st
import plotly.express as px
import pandas as pd
from services.analytics import promo_suggestions, simulate_revenue_curve  # 🔹 新增

def persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)

def show_product_mix_only(tx: pd.DataFrame):
    st.header("🧪 Product Mix (Only)")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    if "Category" not in tx.columns or "Qty" not in tx.columns:
        st.info("Required columns 'Category' and 'Qty' not found.")
        return

    with st.sidebar:
        st.subheader("Filter")
        cats = sorted(tx["Category"].dropna().astype(str).unique().tolist())
        sel = persisting_multiselect("Categories", cats, key="pmxonly_cats")

    df = tx.copy()
    if sel:
        df = df[df["Category"].astype(str).isin(sel)]

    mix = (
        df.groupby("Category", as_index=False)["Qty"]
        .sum()
        .rename(columns={"Qty": "count"})
        .sort_values("count", ascending=False)
    )

    if not mix.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(mix, x="Category", y="count", title="Category Composition (by Qty)"),
                use_container_width=True,
            )
        with c2:
            fig = px.bar(
                mix.sort_values("count", ascending=True),
                x="count",
                y="Category",
                orientation="h",
                title="Category Share",
            )
            fig.update_layout(yaxis=dict(title="Category", automargin=True))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(mix, use_container_width=True)
    else:
        st.info("No data under current filters.")

    # 🔹 新增: Pricing & Promotion - Discount Forecast Suggestions
    st.subheader("💡 Discount Forecast Suggestions")
    sugg = promo_suggestions(tx)
    if not sugg.empty:
        st.table(sugg)
    else:
        st.info("No suggestions available.")

    st.caption("Simulated revenue under different strategies (next 30 days)")
    fut = simulate_revenue_curve(tx)
    if not fut.empty:
        st.plotly_chart(
            px.line(
                fut,
                x="date",
                y=["baseline", "popular_bundle", "popular_slow", "discount_slow"],
                labels={"value": "Revenue", "date": "Date"},
                title="Simulated Revenue Curves"
            ),
            use_container_width=True
        )
    else:
        st.info("Not enough data for revenue simulation.")
