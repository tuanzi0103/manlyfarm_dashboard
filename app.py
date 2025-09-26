import streamlit as st
import pandas as pd
import re

from services.analytics import load_all
from services.db import get_db

# 新增的 Section 模块
from charts.high_level import show_high_level
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory          # 新的库存模块
from charts.product_mix_only import show_product_mix_only # 纯“品类构成”
from charts.customer_segmentation import show_customer_segmentation

# =============== Sidebar ===============
st.sidebar.header("⚙️ Dashboard")

# Section 切换（按你的命名与顺序）
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

# 时间范围
range_choice = st.sidebar.selectbox(
    "Select Time Range",
    ["All", "Past 1 month", "Past 3 months", "Past 6 months", "Past 9 months"]
)
time_from, time_to = None, None
if range_choice != "All":
    months = int(re.search(r"\d+", range_choice).group())
    time_to = pd.Timestamp.today().floor("D")
    time_from = time_to - pd.DateOffset(months=months)

# 数据导入保持原来的交互（沿用你的 ingestion 逻辑）
uploaded_files = st.sidebar.file_uploader(
    "Upload Excel file and import to database",
    type=["xlsx"],
    accept_multiple_files=True
)
enable_fake = st.sidebar.checkbox("Use Faker to complete (FirstName/Surname/Email/Phone)", value=False)

if uploaded_files:
    from services.ingestion import ingest_excel, _sanitize_for_mongo
    for f in uploaded_files:
        try:
            collection_name, inserted_df = ingest_excel(f, enable_fake=enable_fake)
            st.sidebar.success(f"{collection_name} data {f.name} Import successfully to MongoDB ✅ ({len(inserted_df)} lines)")
        except Exception as e:
            st.sidebar.error(f"{f.name} Import Failed ❌: {e}")

# 计量单位维护（保持原样）
st.sidebar.subheader("📏 Add Stocking Unit")
unit_name = st.sidebar.text_input("Unit Name")
unit_value = st.sidebar.number_input("Conversion Base (1 = default)", value=1.0)
if st.sidebar.button("Add Unit"):
    from services.ingestion import _sanitize_for_mongo
    db = get_db()
    doc = _sanitize_for_mongo({"name": unit_name, "value": unit_value})
    db.units.update_one({"name": unit_name}, {"$set": doc}, upsert=True)
    st.sidebar.success(f"Unit {unit_name} has been added/updated ✅")

# 清空数据库（保持原样）
if st.sidebar.button("Clear Database"):
    db = get_db()
    db.transactions.delete_many({})
    db.members.delete_many({})
    db.inventory.delete_many({})
    st.sidebar.warning("All data cleared ❌")

# =============== Main ===============
st.title("📊 Manly Farm Dashboard")

# 按时间范围加载数据（沿用你的 load_all）
tx, mem, inv = load_all(time_from, time_to)

# 路由到各 Section
if section == "Section 1: High Level report":
    show_high_level(tx, mem, inv)

elif section == "Section 2: Sales report by category":
    show_sales_report(tx, inv)

elif section == "Section 3: Inventory":
    show_inventory(tx, inv)

elif section == "Section 4: product mix":
    show_product_mix_only(tx)

elif section == "Section 5: Customers insights":
    # 保留原“客户洞察”的所有逻辑
    show_customer_segmentation(tx, mem)
