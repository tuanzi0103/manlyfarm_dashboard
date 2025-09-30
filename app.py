import streamlit as st
import pandas as pd
import re

from services.analytics import load_all
from services.db import get_db
from services.ingestion import ingest_excel, ingest_csv
from charts.high_level import show_high_level
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory
from charts.product_mix_only import show_product_mix_only
from charts.customer_segmentation import show_customer_segmentation


# === 数据库加载（缓存，可刷新） ===
@st.cache_data(show_spinner=False)
def load_db_cached():
    """加载数据库里的所有数据"""
    return load_all(None, None, get_db())


# =============== Sidebar ===============
st.sidebar.header("⚙️ Dashboard")

# Section 切换
section = st.sidebar.radio(
    "Choose Analysis Perspectives",
    [
        "Section 1: High Level report",
        "Section 2: Sales report by category",
        "Section 3: Inventory",
        "Section 4: product mix",
        "Section 5: Customers insights",
    ],
    index=0
)

# === 上传功能（永远显示） ===
uploaded_files = st.sidebar.file_uploader(
    "📂 Upload data file (CSV/Excel)",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

tx, mem, inv = load_db_cached()  # 默认加载历史数据

if uploaded_files:
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name}..."):
            if f.name.lower().endswith(".xlsx"):
                ingest_excel(f)
            elif f.name.lower().endswith(".csv"):
                ingest_csv(f)
        st.sidebar.success(f"{f.name} imported ✅")

    # 🔄 清缓存并重新加载数据库
    st.cache_data.clear()
    tx, mem, inv = load_all(None, None, get_db())
    st.sidebar.success("✅ Data uploaded and dashboard refreshed with new data")

# === Add Unit 功能 ===
st.sidebar.subheader("➕ Add New Unit")
new_unit = st.sidebar.text_input("Enter new unit name")
if st.sidebar.button("Add Unit"):
    if new_unit:
        db = get_db()
        db.inventory.insert_one({"Unit": new_unit})
        st.sidebar.success(f"✅ Unit '{new_unit}' added successfully")
        st.cache_data.clear()
        tx, mem, inv = load_all(None, None, get_db())  # 立刻刷新
    else:
        st.sidebar.error("⚠️ Please enter a unit name before adding")

# === 清空数据库按钮 ===
if st.sidebar.button("Clear Database"):
    db = get_db()
    db.transactions.delete_many({})
    db.members.delete_many({})
    db.inventory.delete_many({})
    st.sidebar.warning("All data cleared ❌")
    st.session_state["uploaded_history"] = set()
    st.cache_data.clear()
    tx, mem, inv = None, None, None


# =============== Main ===============
st.title("📊 Manly Farm Dashboard")

if (tx is not None and not tx.empty) or (mem is not None and not mem.empty) or (inv is not None and not inv.empty):

    if section == "Section 1: High Level report":
        show_high_level(tx, mem, inv)

    elif section == "Section 2: Sales report by category":
        show_sales_report(tx, inv)

    elif section == "Section 3: Inventory":
        show_inventory(tx, inv)

    elif section == "Section 4: product mix":
        show_product_mix_only(tx)

    elif section == "Section 5: Customers insights":
        show_customer_segmentation(tx, mem)

else:
    st.info("📂 No historical data found in database. Please upload CSV/Excel files from sidebar.")
