import streamlit as st
import pandas as pd
from services.analytics import load_all
from services.db import get_db
from services.ingestion import ingest_excel, ingest_csv
from charts.high_level import show_high_level
# 其他模块
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory
from charts.product_mix_only import show_product_mix_only
from charts.customer_segmentation import show_customer_segmentation
from services.ingestion import ingest_excel, ingest_csv, init_db_from_drive_once
from services.init_db import init_db
from services.ingestion import init_db_from_drive_once
import os
os.environ["WATCHDOG_DISABLE_FILE_WATCH"] = "true"

# ✅ 确保 SQLite 文件存在
init_db()
# ✅ 确保有数据表（如果是空的就从 Drive 拉）
init_db_from_drive_once()


st.set_page_config(page_title="Manly Farm Dashboard", layout="wide")
st.title("📊 Manly Farm Dashboard")


# ✅ 缓存数据库加载
@st.cache_data(show_spinner="loading...")
def load_db_cached(days=365):
    db = get_db()
    return load_all(days=days, db=db)

# === 数据加载 ===
tx, mem, inv = load_db_cached()

# === Sidebar 功能 ===
st.sidebar.header("⚙️ Settings")

# 文件上传（CSV / Excel）
uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    db = get_db()
    for f in uploaded_files:
        if f.name.lower().endswith(".xlsx"):
            ingest_excel(f)
        elif f.name.lower().endswith(".csv"):
            ingest_csv(f)
    st.sidebar.success("✅ Files ingested successfully.")
    load_db_cached.clear()   # 清理缓存，下次刷新自动重算

# === 清空数据库 ===
if st.sidebar.button("🗑️ Clear Database"):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions")
    cur.execute("DELETE FROM inventory")
    cur.execute("DELETE FROM members")
    conn.commit()
    st.sidebar.success("✅ Database cleared!")
    load_db_cached.clear()


# === 单位选择 ===
st.sidebar.subheader("📏 Units")

if inv is not None and not inv.empty and "Unit" in inv.columns:
    units_available = sorted(inv["Unit"].dropna().unique().tolist())
else:
    units_available = ["Gram 1.000", "Kilogram 1.000", "Milligram 1.000"]

db = get_db()
db_units = [u["name"] for u in db.units.find({}, {"_id": 0, "name": 1})]

all_units = sorted(list(set(units_available + db_units)))
unit = st.sidebar.selectbox("Choose unit", all_units)

new_unit = st.sidebar.text_input("Add new unit")
if st.sidebar.button("➕ Add Unit"):
    if new_unit and new_unit not in all_units:
        db.units.insert_one({"name": new_unit})
        st.sidebar.success(f"✅ Added new unit: {new_unit}")
        st.experimental_rerun()

# === Section 选择 ===
section = st.sidebar.radio("📂 Sections", [
    "Section 1: High Level report",
    "Section 2: Sales report by category",
    "Section 3: Inventory",
    "Section 4: product mix",
    "Section 5: Customers insights"
])

# === 主体展示 ===
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
