import pandas as pd
import streamlit as st
from services.db import get_db
from services.analytics import daily_summary_mongo, category_summary_mongo

# ========== 帮助函数 ==========

def _is_transaction_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    return ("date" in cols or "datetime" in cols) and ("net sales" in cols)

def _is_member_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    return ("customer id" in cols or "square customer id" in cols)

def _is_inventory_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    return ("sku" in cols) and (
        ("current quantity vie market & bar" in cols)
        or ("stock on hand" in cols)
        or any(c.startswith("current quantity") for c in cols)
    )

def preprocess_transaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # 合并 Date + Time → Datetime
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    # 去重
    df = df.drop_duplicates()
    return df

def preprocess_member(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "Square Customer ID" in df.columns and "Customer ID" not in df.columns:
        df = df.rename(columns={"Square Customer ID": "Customer ID"})
    df = df.drop_duplicates()
    return df

def preprocess_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # ✅ 兼容库存不同叫法
    if "Stock on Hand" in df.columns and "Current Quantity Vie Market & Bar" not in df.columns:
        df = df.rename(columns={"Stock on Hand": "Current Quantity Vie Market & Bar"})

    for col in df.columns:
        if col.lower().startswith("current quantity") and col != "Current Quantity Vie Market & Bar":
            df = df.rename(columns={col: "Current Quantity Vie Market & Bar"})
            break

    df = df.drop_duplicates()
    return df

# ========== 主导入函数 ==========

def ingest_csv(uploaded_file, enable_fake=False):
    db = get_db()
    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file)
    st.sidebar.info(f"📂 Importing CSV: {filename}")

    try:
        df = pd.read_csv(uploaded_file)
        results = {}

        if _is_transaction_sheet(df):
            df = preprocess_transaction(df)
            if not df.empty:
                db.transactions.insert_many(df.to_dict("records"))
                results["transactions"] = len(df)

        elif _is_member_sheet(df):
            df = preprocess_member(df)
            if not df.empty:
                db.members.insert_many(df.to_dict("records"))
                results["members"] = len(df)

        elif _is_inventory_sheet(df):
            df = preprocess_inventory(df)
            if not df.empty:
                db.inventory.insert_many(df.to_dict("records"))
                results["inventory"] = len(df)

        st.sidebar.success(f"✅ {filename} imported ({len(df)} rows)")

        # ✅ 刷新 summary
        daily_summary_mongo(db, refresh=True)
        category_summary_mongo(db, refresh=True)

        return results

    except Exception as e:
        st.sidebar.error(f"❌ Error importing {filename}: {e}")
        return {}


def ingest_excel(uploaded_file):
    db = get_db()
    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file)
    st.sidebar.info(f"📂 Importing Excel: {filename}")

    try:
        xls = pd.ExcelFile(uploaded_file)
        results = {}

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)

            if _is_transaction_sheet(df):
                df = preprocess_transaction(df)
                if not df.empty:
                    db.transactions.insert_many(df.to_dict("records"))
                    results["transactions"] = len(df)

            elif _is_member_sheet(df):
                df = preprocess_member(df)
                if not df.empty:
                    db.members.insert_many(df.to_dict("records"))
                    results["members"] = len(df)

            elif _is_inventory_sheet(df):
                df = preprocess_inventory(df)
                if not df.empty:
                    db.inventory.insert_many(df.to_dict("records"))
                    results["inventory"] = len(df)

        st.sidebar.success(f"✅ {filename} imported ({sum(results.values())} rows)")

        # ✅ 刷新 summary
        daily_summary_mongo(db, refresh=True)
        category_summary_mongo(db, refresh=True)

        return results

    except Exception as e:
        st.sidebar.error(f"❌ Error importing {filename}: {e}")
        return {}
