import streamlit as st
import plotly.express as px
import pandas as pd
from itertools import combinations

from services.simulator import simulate_combo_revenue


# ----------------- 工具函数 -----------------
def persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)


def _item_col(df: pd.DataFrame) -> str:
    if "Item" in df.columns:
        return "Item"
    if "Item Name" in df.columns:
        return "Item Name"
    return df.columns[0]


def _build_pairs_by_order(df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """基于同一订单统计商品对"""
    df2 = df.copy()

    order_key = None
    for col in ["Order ID", "Receipt ID", "Txn ID", "Transaction ID"]:
        if col in df2.columns:
            order_key = df2[col].astype(str)
            break
    if order_key is None:
        if "Datetime" in df2.columns:
            key = df2["Datetime"].dt.floor("min").astype(str)
            if "Customer ID" in df2.columns:
                key = key + "_" + df2["Customer ID"].astype(str)
            order_key = key
        else:
            order_key = pd.Series(["__one__"] * len(df2), index=df2.index)

    df2 = df2[[item_col]].assign(order_key=order_key)
    pair_counter = {}

    for _, g in df2.groupby("order_key"):
        items = sorted(set(g[item_col].dropna().astype(str)))
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            key = (a, b)
            pair_counter[key] = pair_counter.get(key, 0) + 1

    if not pair_counter:
        return pd.DataFrame(columns=["a", "b", "count"])

    pairs = pd.DataFrame([(k[0], k[1], v) for k, v in pair_counter.items()],
                         columns=["a", "b", "count"]).sort_values("count", ascending=False)
    return pairs


def _strategy_suggestions(df: pd.DataFrame, top_pairs: int = 6, slow_k: int = 6) -> dict:
    """生成三类策略的商品组合"""
    if df.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    item_col = _item_col(df)

    item_cnt = (
        df.groupby(item_col)["Qty"]
        .sum()
        .reset_index(name="qty")
        .sort_values("qty", ascending=False)
    )
    if item_cnt.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    q75 = item_cnt["qty"].quantile(0.75)
    q25 = item_cnt["qty"].quantile(0.25)
    popular_items = item_cnt[item_cnt["qty"] >= q75][item_col].astype(str).tolist()
    slow_items = item_cnt[item_cnt["qty"] <= q25][item_col].astype(str).tolist()

    pairs = _build_pairs_by_order(df, item_col)

    # popular-popular
    pp = []
    if not pairs.empty and popular_items:
        mask = pairs["a"].isin(popular_items) & pairs["b"].isin(popular_items)
        pp = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())

    # popular-slow
    ps = []
    if not pairs.empty and popular_items and slow_items:
        mask = (pairs["a"].isin(popular_items) & pairs["b"].isin(slow_items)) | \
               (pairs["a"].isin(slow_items) & pairs["b"].isin(popular_items))
        ps = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())
    if not ps and popular_items and slow_items:
        for a, b in zip(popular_items[:top_pairs], slow_items[:top_pairs]):
            ps.append(f"{a} + {b}")

    # discount slow
    ds = slow_items[:slow_k]

    return {"popular_popular": pp, "popular_slow": ps, "discount_slow": ds}


# ----------------- 主入口 -----------------
def show_product_mix_only(tx: pd.DataFrame):
    st.header("📊 Product Mix")

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    if "Category" not in tx.columns or "Qty" not in tx.columns:
        st.info("Required columns 'Category' and 'Qty' not found.")
        return

    # ---- 多选搜索框 ----
    cats = sorted(tx["Category"].dropna().astype(str).unique().tolist())
    sel = persisting_multiselect("Search/Filter Categories", cats, key="pmxonly_cats")

    df = tx.copy()
    if sel:
        df = df[df["Category"].astype(str).isin(sel)]
    if df.empty:
        st.info("No data for selected categories.")
        return

    # ---- Category Composition ----
    st.subheader("Category Composition (by Qty)")
    mix = (
        df.groupby("Category", as_index=False)["Qty"]
        .sum()
        .rename(columns={"Qty": "count"})
        .sort_values("count", ascending=False)
    )
    fig_comp = px.bar(
        mix,
        x="Category",
        y="count",
        title="Category Composition (Absolute Count)",
        labels={"count": "Total Quantity"},
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ---- Category Share ----
    st.subheader("Category Share (%)")
    total = mix["count"].sum()
    mix["percent"] = (mix["count"] / total) * 100
    fig_share = px.bar(
        mix,
        x="Category",
        y="percent",
        color="Category",
        title="Category Share (100% stacked)",
        labels={"percent": "Percent (%)"},
    )
    fig_share.update_yaxes(range=[0, 8], ticksuffix="%")  # 🔹 Y轴限制 0–8%
    st.plotly_chart(fig_share, use_container_width=True)

    # ---- Discount Forecast Suggestions ----
    st.subheader("💡 Discount Forecast Suggestions")
    item_col = _item_col(df)
    sugg = _strategy_suggestions(df[[item_col, "Qty", "Datetime", "Customer ID"]]
                                 .dropna(subset=[item_col], how="any"), top_pairs=6, slow_k=6)

    rows = []
    for p in sugg["popular_popular"]:
        rows.append({"strategy": "Bundle popular-popular", "combo": p})
    for p in sugg["popular_slow"]:
        rows.append({"strategy": "Bundle popular-slow", "combo": p})
    for s in sugg["discount_slow"]:
        rows.append({"strategy": "Discount slow mover", "combo": s})
    sugg_df = pd.DataFrame(rows, columns=["strategy", "combo"])

    if sugg_df.empty:
        st.info("No suggestions available based on current filters.")
    else:
        st.table(sugg_df)

    # ---- Simulated Revenue Curves ----
    st.subheader("📈 Simulated Revenue Curves")

    combos = sugg_df["combo"].tolist() if not sugg_df.empty else []

    cols = st.columns([1, 1, 1, 1, 2])
    if cols[0].button("Generate Data 1M", key="gen_mix_1m") and combos:
        st.session_state["pmix_curves"] = simulate_combo_revenue(combos, months=1)
    if cols[1].button("Generate Data 3M", key="gen_mix_3m") and combos:
        st.session_state["pmix_curves"] = simulate_combo_revenue(combos, months=3)
    if cols[2].button("Generate Data 6M", key="gen_mix_6m") and combos:
        st.session_state["pmix_curves"] = simulate_combo_revenue(combos, months=6)
    if cols[3].button("Generate Data 9M", key="gen_mix_9m") and combos:
        st.session_state["pmix_curves"] = simulate_combo_revenue(combos, months=9)
    if cols[4].button("Clear Data", key="clear_mix"):
        st.session_state.pop("pmix_curves", None)

    fut_long = st.session_state.get("pmix_curves", pd.DataFrame())

    # ---- 搜索框 (多选 combo) ----
    if not fut_long.empty:
        options = sorted(fut_long["combo"].unique())
        selected = st.multiselect("Search/Filter Combos", options, default=options, key="combo_filter")
        if selected:
            fut_long = fut_long[fut_long["combo"].isin(selected)]

    if fut_long.empty:
        st.info("No simulated revenue data. Click Generate Data above.")
    else:
        fig_curves = px.line(
            fut_long,
            x="date",
            y="revenue",
            color="combo",
            title="Simulated Revenue Curves by Combo",
            labels={"revenue": "Revenue", "date": "Date", "combo": "Combo"},
        )
        st.plotly_chart(fig_curves, use_container_width=True)
