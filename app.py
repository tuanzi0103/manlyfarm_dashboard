import streamlit as st
import pandas as pd
import re

from services.ingestion import ingest_excel
from services.analytics import load_all
from services.db import get_db
from services.ingestion import _sanitize_for_mongo

# charts
from charts.customer_segmentation import show_customer_segmentation
from charts.product_mix import show_product_mix
from charts.pricing_promotion import show_pricing_promotion
from charts.retention_loyalty import show_retention_loyalty


# ===================
# Sidebar
# ===================
st.sidebar.header("⚙️ Dashboard")

# Analysis direction switch
section = st.sidebar.radio(
    "Choose Analysis Perspectives",
    ["Customer Segmentation & Personalization", "Product Mix & Inventory Optimisation",
     "Pricing & Promotion Strategies", "Customer Retention & Loyalty"],
    index=0
)

# Time range
range_choice = st.sidebar.selectbox(
    "Select Time Range",
    ["All", "Past 1 month", "Past 3 months", "Past 6 months", "Past 9 months"]
)
time_from, time_to = None, None
if range_choice != "All":
    match = re.search(r"\d+", range_choice)
    if match:
        months = int(match.group())
    time_to = pd.Timestamp.today().floor("D")
    time_from = time_to - pd.DateOffset(months=months)

# Import Excel → MongoDB
uploaded_files = st.sidebar.file_uploader(
    "Upload Excel file and import to database",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)
enable_fake = st.sidebar.checkbox("Use Faker to complete (FirstName/Surname/Email/Phone)", value=False)

if uploaded_files:
    for f in uploaded_files:
        try:
            collection_name, inserted_df = ingest_excel(f, enable_fake=enable_fake)
            st.sidebar.success(
                f"{collection_name} data {f.name} imported successfully ✅ ({len(inserted_df)} rows)"
            )
        except Exception as e:
            st.sidebar.error(f"{f.name} import failed ❌: {e}")

# Add stocking unit
st.sidebar.subheader("📏 Add Stocking Unit")
unit_name = st.sidebar.text_input("Unit Name")
unit_value = st.sidebar.number_input("Conversion Base (1 = default)", value=1.0)
if st.sidebar.button("Add Unit"):
    db = get_db()
    doc = _sanitize_for_mongo({"name": unit_name, "value": unit_value})
    db.units.update_one({"name": unit_name}, {"$set": doc}, upsert=True)
    st.sidebar.success(f"Unit {unit_name} has been added/updated ✅")

# Clear database
if st.sidebar.button("Clear Database"):
    db = get_db()
    db.transactions.delete_many({})
    db.members.delete_many({})
    db.inventory.delete_many({})
    st.sidebar.warning("All data cleared ❌")

# ===================
# Main Layout
# ===================
st.title("📊 Manly Farm Dashboard")

# Load data (based on time range, cached)
tx, mem, inv = load_all(time_from, time_to)

if section == "Customer Segmentation & Personalization":
    show_customer_segmentation(tx, mem)
elif section == "Product Mix & Inventory Optimisation":
    show_product_mix(tx, inv)
elif section == "Pricing & Promotion Strategies":
    show_pricing_promotion(tx)
elif section == "Customer Retention & Loyalty":
    show_retention_loyalty(tx, mem)
