import streamlit as st
import plotly.express as px
import pandas as pd
from services.analytics import (
    member_flagged_transactions,
    member_frequency_stats,
    non_member_overview,
    category_counts,
    heatmap_pivot,
    top_categories_for_customer,
    recommend_similar_categories,
    ltv_timeseries_for_customer,
    recommend_bundles_for_customer,
    churn_signals_for_member,
)


def show_customer_segmentation(tx, members):
    st.header("👥 Customer Segmentation & Personalization")

    if tx.empty:
        st.info("No transaction data available.")
        return

    df = member_flagged_transactions(tx)

    # 1) User analysis
    st.subheader("1) User Analysis")
    mode = st.selectbox("Select Target Group", ["Members", "Non-Members"])

    if mode == "Members":
        m = df[df["is_member"]].copy()
        stats = member_frequency_stats(m)

        # 🔹 姓名拼接：检查是否存在列
        if "First Name" in stats.columns and "Surname" in stats.columns:
            stats["Name"] = (stats["First Name"].fillna("") + " " + stats["Surname"].fillna("")).str.strip()
        else:
            stats["Name"] = ""

        show_cols = [c for c in ["Customer ID", "Name", "Phone",
                                 "visits", "last_visit", "net_sales"] if c in stats.columns or c == "Name"]
        q = st.text_input("Search by Customer ID / Name / Phone")
        if q and not stats.empty:
            mask = (
                stats["Customer ID"].astype(str).str.contains(q, case=False, na=False)
                | stats["Name"].astype(str).str.contains(q, case=False, na=False)
                | stats.get("Phone", pd.Series("", index=stats.index)).astype(str).str.contains(q, case=False, na=False)
            )
            stats = stats[mask]

        st.dataframe(stats[show_cols] if show_cols else stats, use_container_width=True)

    else:
        nm = df[~df["is_member"]].copy()
        stats_s = non_member_overview(nm)
        if not stats_s.empty:
            s = stats_s.to_dict()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Foot Traffic", int(s.get("traffic", 0) or 0))
            c2.metric("Product Sales", f"{s.get('Product Sales', 0):.2f}")
            c3.metric("Discounts", f"{s.get('Discounts', 0):.2f}")
            c4.metric("Net Sales", f"{s.get('Net Sales', 0):.2f}")
            c5.metric("Gross Sales", f"{s.get('Gross Sales', 0):.2f}")
        else:
            st.info("No non-member statistics available")

    # 2) User Purchase Preferences
    st.subheader("2) User Purchase Preferences")
    cc = category_counts(tx)
    cat_query_multi = st.multiselect("Search Category", options=cc["Category"].astype(str).unique().tolist())
    if cat_query_multi and not cc.empty:
        cc = cc[cc["Category"].astype(str).isin(cat_query_multi)]

    if not cc.empty:
        st.plotly_chart(px.bar(cc, x="Category", y="count", title="Purchase Quantity by Category"), use_container_width=True)
    else:
        st.info("No category statistics available to display.")

    # 3) Purchase Time Heatmap
    st.subheader("3) Purchase Time Heatmap")
    pv = heatmap_pivot(tx)
    if not pv.empty:
        st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="Blues", title="Shopping Peak Hours"), use_container_width=True)
    else:
        st.info("Not enough time data to generate a heatmap.")

    # 4) Personalized Recommendations
    st.subheader("4) Personalized Recommendations")
    who = st.selectbox("Recommendation Target", ["Member", "Non-Member"], key="rec_mode")
    if who == "Member":
        cid_or_name = st.text_input("Enter Customer ID or Name")
        if cid_or_name:
            sub = df[df["is_member"]].copy()
            if "First Name" in sub.columns and "Surname" in sub.columns:
                sub["Name"] = (sub["First Name"].fillna("") + " " + sub["Surname"].fillna("")).str.strip()
            else:
                sub["Name"] = ""
            match = sub[(sub["Customer ID"].astype(str) == cid_or_name) | (sub["Name"].str.contains(cid_or_name, case=False, na=False))]
            if not match.empty:
                cid = match.iloc[0]["Customer ID"]

                cat_stats = top_categories_for_customer(tx, cid)
                if not cat_stats.empty:
                    st.plotly_chart(
                        px.bar(cat_stats, x="Category", y="qty", title=f"Top Categories for Member {cid_or_name}"),
                        use_container_width=True
                    )

                st.write("Similar Popular Categories (from transactions):")
                rec = recommend_similar_categories(tx, cust_id=cid)
                if not rec.empty:
                    st.plotly_chart(
                        px.bar(rec, x="Category", y="count", title="Popular Categories by Transactions"),
                        use_container_width=True
                    )
    else:
        st.write("General Recommendations:")
        rec = recommend_similar_categories(tx)
        if not rec.empty:
            st.plotly_chart(
                px.bar(rec, x="Category", y="count", title="Popular Categories (Non-Member, Transactions)"),
                use_container_width=True
            )

    # 🔹 Retention & Loyalty
    st.subheader("🔁 Customer Retention & Loyalty")

    mem_tx = tx[tx["is_member"]] if "is_member" in tx.columns else tx.copy()
    tab1, tab2 = st.tabs(["Returning Customers", "Churn Analysis"])

    # Returning Customers
    with tab1:
        st.subheader("Returning Customers (with LTV Forecast)")
        cid_or_name = st.text_input("Enter Customer ID or Name for LTV Forecast")
        if cid_or_name:
            if "First Name" in mem_tx.columns and "Surname" in mem_tx.columns:
                mem_tx["Name"] = (mem_tx["First Name"].fillna("") + " " + mem_tx["Surname"].fillna("")).str.strip()
            else:
                mem_tx["Name"] = ""
            match = mem_tx[(mem_tx["Customer ID"].astype(str) == cid_or_name) | (mem_tx["Name"].str.contains(cid_or_name, case=False, na=False))]
            if not match.empty:
                cid = match.iloc[0]["Customer ID"]
                ltv = ltv_timeseries_for_customer(mem_tx, cid, horizon_days=180)
                if not ltv.empty:
                    st.plotly_chart(
                        px.line(ltv, x="date", y="expected_ltv", title=f"Expected Cumulative LTV (180d) for {cid_or_name}"),
                        use_container_width=True,
                    )
                rec = recommend_bundles_for_customer(mem_tx, cid)
                if not rec.empty:
                    st.table(rec)

    # Churn Analysis
    with tab2:
        st.subheader("Churn Analysis (Members Only)")
        sig = churn_signals_for_member(mem_tx)
        if not sig.empty:
            # 🔹 merge会员信息，保证有姓名和手机号
            enrich_cols = ["Customer ID"]
            if "First Name" in mem_tx.columns:
                enrich_cols.append("First Name")
            if "Surname" in mem_tx.columns:
                enrich_cols.append("Surname")
            if "Phone" in mem_tx.columns:
                enrich_cols.append("Phone")

            sig = sig.merge(mem_tx[enrich_cols].drop_duplicates("Customer ID"),
                            on="Customer ID", how="left")

            # 拼接 Name
            if "First Name" in sig.columns and "Surname" in sig.columns:
                sig["Name"] = (sig["First Name"].fillna("") + " " + sig["Surname"].fillna("")).str.strip()
            else:
                sig["Name"] = ""
            sig["display"] = sig["Name"].replace("", pd.NA).fillna(sig["Customer ID"])

            show_cols = ["Customer ID", "Name", "days_since", "risk_flag"]
            if "Phone" in sig.columns:
                show_cols.insert(2, "Phone")

            fig = px.bar(
                sig,
                x="display",
                y="days_since",
                color="risk_flag",
                title="Days Since Last Purchase (Red = At Risk)",
                labels={"days_since": "Days Since Last Purchase"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sig[show_cols], use_container_width=True)
        else:
            st.info("No churn signals available.")
