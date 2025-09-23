import streamlit as st
import plotly.express as px
from services.analytics import (
    member_flagged_transactions,
    member_frequency_stats,
    non_member_overview,
    category_counts,
    heatmap_pivot,
    top_categories_for_customer,
    recommend_similar_categories,
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

        show_cols = [c for c in ["Customer ID", "First Name", "Surname", "Email", "Phone",
                                 "visits", "last_visit", "net_sales"] if c in stats.columns]
        q = st.text_input("Search by Customer ID / Name / Email / Phone")
        if q and not stats.empty:
            mask = False
            for col in show_cols:
                mask = mask | stats[col].astype(str).str.contains(q, case=False, na=False)
            stats = stats[mask]

        st.dataframe(stats[show_cols] if show_cols else stats, use_container_width=True)

        if not m.empty:
            c1, c2 = st.columns(2)
            with c1:
                agg = m.groupby("Customer ID", as_index=False)["Net Sales"].sum().sort_values("Net Sales", ascending=False).head(20)
                if "First Name" in m.columns:
                    id2name = m.groupby("Customer ID")[["First Name", "Surname"]].agg(
                        lambda x: x.dropna().iloc[0] if x.dropna().any() else ""
                    ).reset_index()
                    agg = agg.merge(id2name, on="Customer ID", how="left")
                    agg["display"] = agg.apply(
                        lambda r: f"{r['First Name']} {r['Surname']}".strip() or r["Customer ID"], axis=1
                    )
                    fig = px.bar(agg, x="display", y="Net Sales", title="Top 20 Members by Total Net Sales")
                else:
                    fig = px.bar(agg, x="Customer ID", y="Net Sales", title="Top 20 Members by Total Net Sales")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                cnt = m.groupby("Customer ID", as_index=False)["Datetime"].count().rename(
                    columns={"Datetime": "visits"}
                ).sort_values("visits", ascending=False).head(20)

                # 🔹 新增：给 visit frequency 也加上名字映射
                if "First Name" in m.columns:
                    id2name = m.groupby("Customer ID")[["First Name", "Surname"]].agg(
                        lambda x: x.dropna().iloc[0] if x.dropna().any() else ""
                    ).reset_index()
                    cnt = cnt.merge(id2name, on="Customer ID", how="left")
                    cnt["display"] = cnt.apply(
                        lambda r: f"{r['First Name']} {r['Surname']}".strip() or r["Customer ID"], axis=1
                    )
                    fig2 = px.bar(cnt, x="display", y="visits", title="Top 20 Members by Visit Frequency")
                else:
                    fig2 = px.bar(cnt, x="Customer ID", y="visits", title="Top 20 Members by Visit Frequency")

                st.plotly_chart(fig2, use_container_width=True)

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
    cat_query = st.text_input("Search Category")
    if cat_query and not cc.empty:
        cc = cc[cc["Category"].astype(str).str.contains(cat_query, case=False, na=False)]
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
        cid = st.text_input("Enter Customer ID")
        if cid:
            # 🔹 显示该会员消费最多的类别
            cat_stats = top_categories_for_customer(tx, cid)
            if not cat_stats.empty:
                st.plotly_chart(
                    px.bar(cat_stats, x="Category", y="qty", title=f"Top Categories for Member {cid}"),
                    use_container_width=True
                )
            else:
                st.info("No category data available for this member.")

            # ✅ 修复：使用 transaction 的 Category，而不是 inventory
            st.write("Similar Popular Categories (from transactions):")
            rec = recommend_similar_categories(tx, cust_id=cid)
            if not rec.empty:
                st.plotly_chart(
                    px.bar(rec, x="Category", y="count", title="Popular Categories by Transactions"),
                    use_container_width=True
                )
    else:
        st.write("General Recommendations:")
        # ✅ 修复：同样用 transaction 的 Category
        rec = recommend_similar_categories(tx)
        if not rec.empty:
            st.plotly_chart(
                px.bar(rec, x="Category", y="count", title="Popular Categories (Non-Member, Transactions)"),
                use_container_width=True
            )
        else:
            st.info("No general recommendations available.")
