import pandas as pd
import hashlib
from io import BytesIO
from faker import Faker
import re
import datetime
from .db import get_db
from pymongo import UpdateOne  # ✅ 新增

fake = Faker()


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def compute_row_hash(row: pd.Series) -> str:
    row_str = "|".join([str(x) for x in row.values])
    return hashlib.md5(row_str.encode()).hexdigest()


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
    sheet_match = False
    if sheet_name:
        low = sheet_name.lower()
        if "export" in low:
            sheet_match = True
        if re.search(r"\d{8}", sheet_name):
            sheet_match = True
    return col_match or sheet_match


def _read_sheet_with_header_fallback(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df0 = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    cols0 = pd.Index(df0.columns)
    if (cols0.astype(str).str.startswith("Unnamed").mean() > 0.6) or cols0.isnull().any():
        df1 = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        return df1
    return df0


def preprocess_transactions(df: pd.DataFrame, enable_fake: bool) -> pd.DataFrame:
    df = clean_headers(df)
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

    df["_row_hash"] = df.apply(compute_row_hash, axis=1)

    if enable_fake:
        for idx, row in df.iterrows():
            if "First Name" in df.columns and pd.isna(row.get("First Name")):
                df.at[idx, "First Name"] = fake.first_name()
            if "Surname" in df.columns and pd.isna(row.get("Surname")):
                df.at[idx, "Surname"] = fake.last_name()
            if "Email" in df.columns and pd.isna(row.get("Email")):
                df.at[idx, "Email"] = fake.email()
            if "Phone" in df.columns and pd.isna(row.get("Phone")):
                df.at[idx, "Phone"] = fake.phone_number()
    return df


def preprocess_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_headers(df)
    df["_row_hash"] = df.apply(compute_row_hash, axis=1)
    return df


def preprocess_members(df: pd.DataFrame, enable_fake: bool) -> pd.DataFrame:
    df = clean_headers(df)
    df["_row_hash"] = df.apply(compute_row_hash, axis=1)
    if "Square Customer ID" in df.columns and "Customer ID" not in df.columns:
        df = df.rename(columns={"Square Customer ID": "Customer ID"})
    if enable_fake:
        for idx, row in df.iterrows():
            if "First Name" in df.columns and pd.isna(row.get("First Name")):
                df.at[idx, "First Name"] = fake.first_name()
            if "Surname" in df.columns and pd.isna(row.get("Surname")):
                df.at[idx, "Surname"] = fake.last_name()
            if "Email" in df.columns and pd.isna(row.get("Email")):
                df.at[idx, "Email"] = fake.email()
            if "Phone" in df.columns and pd.isna(row.get("Phone")):
                df.at[idx, "Phone"] = fake.phone_number()
    return df


def _sanitize_for_mongo(doc: dict) -> dict:
    out = {}
    for k, v in doc.items():
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except Exception:
            pass
        if isinstance(v, pd.Timestamp):
            if pd.isna(v):
                out[k] = None
            else:
                try:
                    if v.tz is not None:
                        out[k] = v.tz_convert(None).to_pydatetime()
                    else:
                        out[k] = v.to_pydatetime()
                except Exception:
                    try:
                        out[k] = v.tz_localize(None).to_pydatetime()
                    except Exception:
                        out[k] = v.to_pydatetime()
            continue
        if isinstance(v, datetime.time):
            out[k] = v.strftime("%H:%M:%S")
            continue
        out[k] = v
    return out


def ingest_excel(uploaded_file, enable_fake=False):
    db = get_db()
    filename = uploaded_file.name
    xls = pd.ExcelFile(BytesIO(uploaded_file.read()))

    inserted_counts = {"transactions": 0, "inventory": 0, "members": 0}
    last_collection = None

    for sheet in xls.sheet_names:
        df_raw = _read_sheet_with_header_fallback(xls, sheet_name=sheet)
        df = clean_headers(df_raw)

        # ---- Transactions ----
        if _is_transaction_sheet(df):
            df2 = preprocess_transactions(df, enable_fake)
            col = db.transactions
            ops = []
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                ops.append(UpdateOne({"_row_hash": _hash}, {"$set": doc}, upsert=True))
            if ops:
                col.bulk_write(ops, ordered=False)  # ✅ 批量写入
            inserted_counts["transactions"] += len(df2)
            last_collection = col.name
            continue

        # ---- Inventory ----
        if _is_inventory_sheet(df):
            df2 = preprocess_inventory(df)
            col = db.inventory
            ops = []
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                ops.append(UpdateOne({"_row_hash": _hash}, {"$set": doc}, upsert=True))
            if ops:
                col.bulk_write(ops, ordered=False)  # ✅ 批量写入
            inserted_counts["inventory"] += len(df2)
            last_collection = col.name

            for colname in df2.columns:
                if colname.strip().lower() == "unit and precision":
                    units_col = db.units
                    unit_ops = []
                    for u in df2[colname].dropna().unique():
                        doc = _sanitize_for_mongo({"name": str(u).strip(), "value": 1.0})
                        unit_ops.append(UpdateOne({"name": doc["name"]}, {"$set": doc}, upsert=True))
                    if unit_ops:
                        units_col.bulk_write(unit_ops, ordered=False)  # ✅ 批量写入
                    break
            continue

        # ---- Members ----
        if _is_member_sheet(df, sheet_name=sheet):
            df2 = preprocess_members(df, enable_fake)
            col = db.members
            ops = []
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                ops.append(UpdateOne({"_row_hash": _hash}, {"$set": doc}, upsert=True))
            if ops:
                col.bulk_write(ops, ordered=False)  # ✅ 批量写入
            inserted_counts["members"] += len(df2)
            last_collection = col.name
            continue

    if sum(inserted_counts.values()) == 0:
        raise ValueError(
            f"No valid sheet detected for import. "
            f"Please ensure at least one sheet contains typical transaction/inventory/member columns. "
            f"File: {filename}, Sheets: {xls.sheet_names}"
        )

    return last_collection or "transactions", pd.DataFrame({"inserted": [inserted_counts]})
