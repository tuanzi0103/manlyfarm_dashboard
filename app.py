# app.py
import os
import streamlit as st
import pandas as pd
import sqlite3
from services.analytics import load_all
from services.db import get_db
from services.ingestion import ingest_excel, ingest_csv, init_db_from_drive_once
from charts.high_level import show_high_level
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory
from charts.product_mix_only import show_product_mix_only
from charts.customer_segmentation import show_customer_segmentation
from init_db import init_db

os.environ["WATCHDOG_DISABLE_FILE_WATCH"] = "true"

# 初始化数据库
init_db()
init_db_from_drive_once()

st.set_page_config(page_title="Manly Farm Dashboard", layout="wide")
st.title("📊 Manly Farm Dashboard")

# === 强制刷新机制 ===
if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = 0


# 直接数据加载函数，不使用缓存
def load_fresh_data():
    """直接从数据库加载最新数据"""
    try:
        db = get_db()
        tx, mem, inv = load_all(db=db)

        # 立即关闭数据库连接
        if hasattr(db, 'close'):
            db.close()

        return tx, mem, inv
    except Exception as e:
        st.error(f"Wrong Data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# 加载数据
tx, mem, inv = load_fresh_data()

# === Sidebar ===
st.sidebar.header("⚙️ Settings")

# 文件上传
uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.force_refresh}"
)

# 处理上传文件
if uploaded_files:
    processed_files = []

    for file in uploaded_files:
        try:
            if file.name.lower().endswith('.xlsx'):
                result = ingest_excel(file)
            elif file.name.lower().endswith('.csv'):
                result = ingest_csv(file)
            else:
                st.sidebar.warning(f"skip {file.name}")
                continue

            if result:
                processed_files.append(file.name)
                st.sidebar.success(f"✅ {file.name}")
            else:
                st.sidebar.error(f"❌ {file.name}")

        except Exception as e:
            st.sidebar.error(f"❌ {file.name}: {str(e)}")

    if processed_files:
        st.sidebar.success(f"✅ Processed {len(processed_files)} files")

        # 强制刷新数据
        st.session_state.force_refresh += 1
        st.rerun()

# Section 选择
section = st.sidebar.radio("📂 Sections", [
    "Section 1: High Level report",
    "Section 2: Sales report by category",
    "Section 3: Inventory",
    "Section 4: product mix",
    "Section 5: Customers insights"
])


if tx is not None and not tx.empty and 'Datetime' in tx.columns:
    latest_loaded = tx['Datetime'].max() if pd.notna(tx['Datetime']).any() else 'N/A'

# 主体展示
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