import pandas as pd
import hashlib
from io import BytesIO
from faker import Faker
import re
import datetime
from .db import get_db

fake = Faker()


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    # 去除全是 Unnamed 的列
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

    # 1) 原始逻辑：典型列
    col_match = ("customer id" in cols) or ({"first name", "surname", "email", "phone"} <= cols)

    # 2) 新增逻辑：sheet 名含 export 或 日期（8位数字如 20250916）
    sheet_match = False
    if sheet_name:
        low = sheet_name.lower()
        if "export" in low:
            sheet_match = True
        if re.search(r"\d{8}", sheet_name):
            sheet_match = True

    return col_match or sheet_match


def _read_sheet_with_header_fallback(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Square 导出常见现象：第一行为空或为说明行，导致读到大量 Unnamed 列。
    策略：先按 header=0 读取；若发现列名几乎全是 Unnamed，则自动用 header=1 再读一次。
    """
    df0 = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    cols0 = pd.Index(df0.columns)
    if (cols0.astype(str).str.startswith("Unnamed").mean() > 0.6) or cols0.isnull().any():
        # 兜底：从第二行作为表头
        df1 = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        return df1
    return df0


def preprocess_transactions(df: pd.DataFrame, enable_fake: bool) -> pd.DataFrame:
    df = clean_headers(df)

    # 合并 Date + Time → Datetime
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

    # 规范列名
    if "Qty" not in df.columns and "Quantity" in df.columns:
        df = df.rename(columns={"Quantity": "Qty"})

    # ✅ 强制只保留 transactions 的 Item，而不是 inventory 的 "Item Name + Variation Name"
    if "Item" in df.columns:
        df["Item"] = df["Item"].astype(str).str.strip()
    elif "Item Name" in df.columns:
        df = df.rename(columns={"Item Name": "Item"})  # 兜底，但不拼接 Variation
    # 不要引入 Variation Name，否则会变成带单位的 inventory 风格

    # 去重键
    df["_row_hash"] = df.apply(compute_row_hash, axis=1)

    # Faker 补全（可选）
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

    # Square Customer ID → Customer ID
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


# ---- 新增：通用清洗（保证能安全写入 Mongo）----
def _sanitize_for_mongo(doc: dict) -> dict:
    out = {}
    for k, v in doc.items():
        # pandas 的缺失值统一转 None
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except Exception:
            pass

        # pandas.Timestamp / NaT / 带时区的时间
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

        # datetime.time → "HH:MM:SS"
        if isinstance(v, datetime.time):
            out[k] = v.strftime("%H:%M:%S")
            continue

        out[k] = v
    return out


def ingest_excel(uploaded_file, enable_fake=False):
    """
    自动识别多 sheet 类型并导入：
    - 若第一行为空/说明行，将自动从第二行读表头（兼容 Square 导出）
    - 识别 transactions / inventory / members 三类
    """
    db = get_db()
    filename = uploaded_file.name
    xls = pd.ExcelFile(BytesIO(uploaded_file.read()))

    inserted_counts = {"transactions": 0, "inventory": 0, "members": 0}
    last_collection = None

    for sheet in xls.sheet_names:
        df_raw = _read_sheet_with_header_fallback(xls, sheet_name=sheet)
        df = clean_headers(df_raw)

        if _is_transaction_sheet(df):
            df2 = preprocess_transactions(df, enable_fake)
            col = db.transactions
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())  # ★ 安全清洗
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                col.update_one({"_row_hash": _hash}, {"$set": doc}, upsert=True)
            inserted_counts["transactions"] += len(df2)
            last_collection = col.name
            continue

        if _is_inventory_sheet(df):
            df2 = preprocess_inventory(df)
            col = db.inventory
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())  # ★ 安全清洗
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                col.update_one({"_row_hash": _hash}, {"$set": doc}, upsert=True)
            inserted_counts["inventory"] += len(df2)
            last_collection = col.name

            # 🔹 新增逻辑：存储 Unit and Precision 到 units 集合（忽略大小写匹配）
            for colname in df2.columns:
                if colname.strip().lower() == "unit and precision":
                    units_col = db.units
                    for u in df2[colname].dropna().unique():
                        doc = _sanitize_for_mongo({"name": str(u).strip(), "value": 1.0})
                        units_col.update_one({"name": doc["name"]}, {"$set": doc}, upsert=True)
                    break

            continue

        if _is_member_sheet(df, sheet_name=sheet):
            df2 = preprocess_members(df, enable_fake)
            col = db.members
            for _, row in df2.iterrows():
                doc = _sanitize_for_mongo(row.to_dict())  # ★ 安全清洗
                _hash = doc.get("_row_hash") or compute_row_hash(row)
                doc["_row_hash"] = _hash
                col.update_one({"_row_hash": _hash}, {"$set": doc}, upsert=True)
            inserted_counts["members"] += len(df2)
            last_collection = col.name
            continue

    if sum(inserted_counts.values()) == 0:
        raise ValueError(
            f"No valid sheet detected for import. "
            f"Please ensure at least one sheet contains typical transaction/inventory/member columns. "
            f"File: {filename}, Sheets: {xls.sheet_names}"
        )

    # 返回最近一次导入的集合名与计数（与现有 UI 兼容）
    return last_collection or "transactions", pd.DataFrame({"inserted": [inserted_counts]})
