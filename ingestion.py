import pandas as pd
from io import BytesIO
from faker import Faker
import re
import datetime
from .db import get_db

fake = Faker()
import streamlit as st

# === 类型识别 ===
def _is_transaction_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    has_time = {"date", "time"} <= cols or "datetime" in cols
    has_item_qty = ("item" in cols or "sku" in cols) and ("qty" in cols or "quantity" in cols)
    has_sales_cols = any(c in cols for c in ["net sales", "gross sales", "discounts", "product sales"])
    return (has_time and has_item_qty) or has_sales_cols

def _is_inventory_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    return ("sku" in cols) and (
        ("stock on hand" in cols)
        or ("stock-by equivalent" in cols)
        or any(c.startswith("current quantity") for c in cols)
    )

def _is_member_sheet(df: pd.DataFrame, sheet_name: str = "") -> bool:
    cols = set(df.columns.str.lower())
    col_match = ("customer id" in cols) or ({"first name", "surname", "email", "phone"} <= cols)
    return col_match or "member" in sheet_name.lower()

def _read_sheet_with_header_fallback(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df0 = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    cols0 = pd.Index(df0.columns)
    if (cols0.astype(str).str.startswith("Unnamed").mean() > 0.6) or cols0.isnull().any():
        df1 = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        return df1
    return df0

# === 预处理函数 ===
def preprocess_transactions(df: pd.DataFrame, enable_fake: bool) -> pd.DataFrame:
    from services.analytics import _to_numeric

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "Date" in df.columns and "Time" in df.columns:
        df["Time"] = df["Time"].apply(
            lambda x: x.strftime("%H:%M:%S") if isinstance(x, datetime.time) else str(x)
        )
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    if "Qty" not in df.columns and "Quantity" in df.columns:
        df = df.rename(columns={"Quantity": "Qty"})
    if "Item" in df.columns:
        df["Item"] = df["Item"].astype(str).str.strip()
    elif "Item Name" in df.columns:
        df = df.rename(columns={"Item Name": "Item"})

    # 转换数值
    for col in ["Net Sales", "Gross Sales", "Qty"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # ✅ 去重
    df = df.drop_duplicates()
    return df


def preprocess_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # ✅ 去重
    df = df.drop_duplicates()
    return df


def preprocess_members(df: pd.DataFrame, enable_fake: bool) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "Square Customer ID" in df.columns and "Customer ID" not in df.columns:
        df = df.rename(columns={"Square Customer ID": "Customer ID"})
    # ✅ 去重
    df = df.drop_duplicates()
    return df

# === Mongo 写入（批量） ===
BATCH_SIZE = 5000

def _insert_many_safe(col, docs):
    """分批批量写入（不检测重复，直接插入）"""
    if not docs:
        return
    for i in range(0, len(docs), BATCH_SIZE):
        col.insert_many(docs[i:i+BATCH_SIZE], ordered=False)

# === CSV 导入 ===
def ingest_csv(uploaded_file, enable_fake=False):
    db = get_db()
    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file)
    st.sidebar.info(f"📂 Importing CSV: {filename}")

    try:
        df = pd.read_csv(uploaded_file)
        results = {}

        if _is_transaction_sheet(df):
            df2 = preprocess_transactions(df, enable_fake)
            needed_cols = ["Datetime", "Item", "Category", "Net Sales", "Gross Sales", "Qty", "Customer ID"]
            df2 = df2[[c for c in needed_cols if c in df2.columns]]
            _insert_many_safe(db.transactions, df2.to_dict("records"))
            results["transactions"] = df2

        elif _is_inventory_sheet(df):
            df2 = preprocess_inventory(df)
            needed_cols = ["SKU", "Item", "Qty", "Net Sales", "Current Quantity Vie Market & Bar"]
            df2 = df2[[c for c in needed_cols if c in df2.columns]]
            _insert_many_safe(db.inventory, df2.to_dict("records"))
            results["inventory"] = df2

        elif _is_member_sheet(df):
            df2 = preprocess_members(df, enable_fake)
            needed_cols = ["Customer ID", "First Name", "Surname", "Email", "Phone"]
            df2 = df2[[c for c in needed_cols if c in df2.columns]]
            _insert_many_safe(db.members, df2.to_dict("records"))
            results["members"] = df2

        else:
            results["unknown"] = df

        st.sidebar.success(f"✅ {filename} imported ({len(df)} rows)")
        return results

    except Exception as e:
        st.sidebar.error(f"❌ {filename} import failed: {e}")
        return {}

# === Excel 导入 ===
def ingest_excel(uploaded_file, enable_fake=False):
    db = get_db()
    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file)
    st.sidebar.info(f"📂 Importing Excel: {filename}")

    try:
        xls = pd.ExcelFile(BytesIO(uploaded_file.read()))
        results = {}

        for sheet in xls.sheet_names:
            df_raw = _read_sheet_with_header_fallback(xls, sheet_name=sheet)
            df = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

            if _is_transaction_sheet(df):
                df2 = preprocess_transactions(df, enable_fake)
                needed_cols = ["Datetime", "Item", "Category", "Net Sales", "Gross Sales", "Qty", "Customer ID"]
                df2 = df2[[c for c in needed_cols if c in df2.columns]]
                _insert_many_safe(db.transactions, df2.to_dict("records"))
                results["transactions"] = results.get("transactions", pd.DataFrame()).append(df2, ignore_index=True)
                continue

            if _is_inventory_sheet(df):
                df2 = preprocess_inventory(df)
                needed_cols = ["SKU", "Item", "Qty", "Net Sales", "Current Quantity Vie Market & Bar"]
                df2 = df2[[c for c in needed_cols if c in df2.columns]]
                _insert_many_safe(db.inventory, df2.to_dict("records"))
                results["inventory"] = results.get("inventory", pd.DataFrame()).append(df2, ignore_index=True)
                continue

            if _is_member_sheet(df, sheet_name=sheet):
                df2 = preprocess_members(df, enable_fake)
                needed_cols = ["Customer ID", "First Name", "Surname", "Email", "Phone"]
                df2 = df2[[c for c in needed_cols if c in df2.columns]]
                _insert_many_safe(db.members, df2.to_dict("records"))
                results["members"] = results.get("members", pd.DataFrame()).append(df2, ignore_index=True)
                continue

        st.sidebar.success(f"✅ {filename} imported")
        return results

    except Exception as e:
        st.sidebar.error(f"❌ {filename} import failed: {e}")
        return {}
