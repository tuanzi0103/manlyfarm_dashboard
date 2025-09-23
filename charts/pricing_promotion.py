import streamlit as st
import plotly.express as px
from services.analytics import discount_breakdown, promo_suggestions, simulate_revenue_curve
import pandas as pd   # 👈 Added


def show_pricing_promotion(tx):
    st.header("💰 Pricing & Promotion Strategies")

    if tx.empty:
        st.info("No transaction data available")
        return

    # 1) Effectiveness of Discount Policies
    st.subheader("1) Effectiveness of Discount Policies")
    bar_df, line_df = discount_breakdown(tx)

    if bar_df.empty and line_df.empty:
        st.info("No discount usage records found in the current data.")
    else:
        if not bar_df.empty:
            st.plotly_chart(
                px.bar(bar_df, x="type", y="buyers", title="Number of Buyers by Discount Type"),
                use_container_width=True
            )
        if not line_df.empty:
            st.plotly_chart(
                px.line(line_df, x="metric", y=["members", "coupons"], markers=True,
                        title="Sales Performance by Discount Type"),
                use_container_width=True
            )

    # 2) Discount Forecast Suggestions + Revenue Curve Simulation
    st.subheader("2) Discount Forecast Suggestions")
    recs = promo_suggestions(tx)
    if not recs.empty:
        st.table(recs)
    else:
        st.info("No suggestions available.")

    st.markdown("**Forecast: Revenue Curve for the Next 30 Days (Trend + Seasonality + Confidence Interval)**")
    curve = simulate_revenue_curve(tx, days=30)
    if not curve.empty:
        y_cols = [c for c in ["baseline", "popular_bundle", "popular_slow", "discount_slow"] if c in curve.columns]

        # Plot forecast curve
        fig = px.line(curve, x="date", y=y_cols, title="Simulated Revenue Curve")

        # Add confidence interval shading
        if "lower_ci" in curve.columns and "upper_ci" in curve.columns:
            fig.add_traces([
                dict(
                    type="scatter",
                    x=pd.concat([curve["date"], curve["date"][::-1]]),
                    y=pd.concat([curve["upper_ci"], curve["lower_ci"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95% CI"
                )
            ])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data to generate a revenue curve.")
