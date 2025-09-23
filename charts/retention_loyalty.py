import streamlit as st
import plotly.express as px
import pandas as pd
from services.analytics import (
    ltv_timeseries_for_customer,
    recommend_bundles_for_customer,
    churn_signals_for_member,
)


def _display_name_for_cid(cid: str, tx: pd.DataFrame, members: pd.DataFrame) -> str:
    """
    Return the display name (First Name + Surname) for a Customer ID.
    Falls back to ID if not found.
    """
    # First check in members
    if members is not None and not members.empty:
        mem = members.copy()
        if "Square Customer ID" in mem.columns and "Customer ID" not in mem.columns:
            mem = mem.rename(columns={"Square Customer ID": "Customer ID"})
        mask = mem.get("Customer ID", pd.Series([], dtype=str)).astype(str) == str(cid)
        if mask.any():
            first = mem.loc[mask, "First Name"].dropna().astype(str).str.strip().head(1)
            last = mem.loc[mask, "Surname"].dropna().astype(str).str.strip().head(1)
            full = (first.iloc[0] if len(first) else "") + " " + (last.iloc[0] if len(last) else "")
            full = full.strip()
            if full:
                return full

    # Then check in transactions
    if tx is not None and not tx.empty and "Customer ID" in tx.columns:
        sub = tx[tx["Customer ID"].astype(str) == str(cid)]
        if not sub.empty:
            first = sub.get("First Name")
            last = sub.get("Surname")
            full = ((first.dropna().astype(str).str.strip().head(1).iloc[0] if first is not None and first.dropna().any() else "") +
                    " " +
                    (last.dropna().astype(str).str.strip().head(1).iloc[0] if last is not None and last.dropna().any() else ""))
            full = full.strip()
            if full:
                return full

    return str(cid)


def show_retention_loyalty(tx, members):
    st.header("🔁 Customer Retention & Loyalty")

    if tx.empty:
        st.info("No data available")
        return

    mem_tx = tx[tx["is_member"]] if "is_member" in tx.columns else tx.copy()
    tab1, tab2 = st.tabs(["Returning Customers", "Churn Analysis"])

    # === Returning Customers ===
    with tab1:
        st.subheader("Returning Customers (with LTV Forecast)")

        q = st.text_input("Search Customer ID / Name / Email / Phone")
        sub = mem_tx.copy()
        if q:
            mask = False
            for col in ["Customer ID", "First Name", "Surname", "Email", "Phone"]:
                if col in sub.columns:
                    mask = mask | sub[col].astype(str).str.contains(q, case=False, na=False)
            sub = sub[mask]

        if not sub.empty:
            sub = sub.sort_values(["Customer ID", "Datetime"])
            sub["prev"] = sub.groupby("Customer ID")["Datetime"].shift()
            sub["interval_days"] = (sub["Datetime"] - sub["prev"]).dt.days
            show = sub["interval_days"].dropna()
            if not show.empty:
                st.plotly_chart(
                    px.histogram(show, x="interval_days", title="Purchase Interval Distribution"),
                    use_container_width=True,
                )
            else:
                st.info("Each matched user has fewer than two purchases, cannot calculate interval distribution.")

        auto_cid = None
        if q and "Customer ID" in sub.columns:
            ids = sub["Customer ID"].astype(str).unique().tolist()
            if len(ids) == 1:
                auto_cid = ids[0]

        cid = st.text_input("Enter Customer ID for LTV Forecast", value=auto_cid or "")
        if cid:
            ltv = ltv_timeseries_for_customer(mem_tx, cid, horizon_days=180)
            if not ltv.empty:
                title_name = _display_name_for_cid(cid, mem_tx, members)
                st.plotly_chart(
                    px.line(ltv, x="date", y="expected_ltv", title=f"{title_name} Expected Cumulative LTV over Next 180 Days"),
                    use_container_width=True,
                )
            else:
                st.info("This user does not have enough history for an LTV forecast.")

            rec = recommend_bundles_for_customer(mem_tx, cid)
            if not rec.empty:
                st.table(rec)

        if not sub.empty:
            info_cols = [
                c
                for c in ["Customer ID", "First Name", "Surname", "Email", "Phone"]
                if c in sub.columns
            ]
            if info_cols:
                st.dataframe(sub[info_cols].drop_duplicates(), use_container_width=True)

    # === Churn Analysis ===
    with tab2:
        st.subheader("Churn Analysis (Members Only)")
        sig = churn_signals_for_member(mem_tx)
        if sig.empty:
            st.info("No churn signals available.")
            return

        # ---- Merge with Square Customer ID if needed ----
        if members is not None and not members.empty:
            mem = members.copy()
            if "Square Customer ID" in mem.columns:
                mem = mem.rename(columns={"Square Customer ID": "Customer ID"})
            keep_cols = [c for c in ["Customer ID", "First Name", "Surname", "Email", "Phone"] if c in mem.columns]
            if "Customer ID" in keep_cols:
                sig = sig.merge(mem[keep_cols], on="Customer ID", how="left")

        # Construct full name
        if "First Name" in sig.columns or "Surname" in sig.columns:
            sig["full_name"] = (
                sig.get("First Name", "").fillna("").astype(str).str.strip()
                + " "
                + sig.get("Surname", "").fillna("").astype(str).str.strip()
            ).str.strip()
            sig["full_name"] = sig["full_name"].replace("", "Unknown User")
        else:
            sig["full_name"] = sig["Customer ID"].astype(str)

        # Ensure Email / Phone columns exist (for hover display)
        for c in ["Email", "Phone"]:
            if c not in sig.columns:
                sig[c] = ""

        # --- New logic: filter out unknown users ---
        sig = sig[sig["full_name"] != "Unknown User"]

        # Search
        q2 = st.text_input("Search Customer ID / Name / Email / Phone", key="churn_q")
        view = mem_tx.copy()
        if q2:
            mask = False
            for col in ["Customer ID", "First Name", "Surname", "Email", "Phone"]:
                if col in view.columns:
                    mask = mask | view[col].astype(str).str.contains(q2, case=False, na=False)
            ids = view[mask]["Customer ID"].astype(str).unique().tolist()
            sig = sig[sig["Customer ID"].astype(str).isin(ids)]

        # --- Fix hover: show actual days + Email / Phone ---
        custom_cols = ["Customer ID", "Email", "Phone"]
        fig = px.bar(
            sig,
            x="full_name",
            y="days_since",
            color="risk_flag",
            custom_data=custom_cols,
            title="Days Since Last Purchase (Red = At Risk)",
            labels={"full_name": "Member Name", "days_since": "Days Since Last Purchase"},
        )
        fig.update_traces(
            hovertemplate=(
                "Name: %{x}"
                "<br>Customer ID: %{customdata[0]}"
                "<br>Days Since Last Purchase: %{y}"
                "<br>Email: %{customdata[1]}"
                "<br>Phone: %{customdata[2]}"
                "<extra></extra>"
            )
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sig.sort_values(["risk_flag", "days_since"], ascending=[False, False]),
            use_container_width=True,
        )
