# services/ingestion.py
import os
import re
import tempfile
from io import BytesIO
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from services.db import get_db

# === Google Drive 相关 ===
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

FOLDER_ID = "1lZGE0DkgKyox1HbBzuhZ-oypDF478jBj"

# ✅ 全局缓存 drive 实例
_drive_instance = None


def get_drive():
    global _drive_instance
    if _drive_instance is not None:
        return _drive_instance

    gauth = GoogleAuth()

    # ✅ 设置 OAuth 为 offline，确保 refresh_token 存下来
    gauth.settings['oauth_scope'] = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.metadata"
    ]
    gauth.settings['get_refresh_token'] = True

    token_path = "credentials.json"
    if os.path.exists(token_path):
        gauth.LoadCredentialsFile(token_path)

    if gauth.credentials is None:
        # 第一次认证：加 offline
        gauth.LocalWebserverAuth()  # 这里会弹浏览器
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile(token_path)
    _drive_instance = GoogleDrive(gauth)
    return _drive_instance


def upload_file_to_drive(local_path: str, remote_name: str):
    """把本地文件上传到指定的 Google Drive 文件夹。"""
    try:
        drive = get_drive()
        f = drive.CreateFile({'title': remote_name, 'parents': [{'id': FOLDER_ID}]})
        f.SetContentFile(local_path)
        f.Upload()
    except Exception as e:
        # 上传失败不影响主流程
        pass


def download_file_from_drive(file_id, local_path):
    drive = get_drive()
    f = drive.CreateFile({'id': file_id})
    f.GetContentFile(local_path)


# --------------- 工具函数 ---------------

def _fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """若第一行是 Unnamed，多数是多行表头；把第二行提为表头。"""
    if len(df.columns) and all(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    return df


def _to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def _extract_date_from_filename(name: str):
    """从文件名中提取 YYYY-MM-DD"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    return None


def _extract_date_range_from_filename(name: str):
    """从文件名中提取日期范围，返回 (start_date, end_date)"""
    # 匹配格式如 2024-08-18_to_2024-09-18 或 2024-08-18-2024-09-18
    patterns = [
        r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})',
        r'(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})',
        r'(\d{4}\.\d{2}\.\d{2})_to_(\d{4}\.\d{2}\.\d{2})',
        r'(\d{4}\.\d{2}\.\d{2})-(\d{4}\.\d{2}\.\d{2})'
    ]

    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            try:
                start_date = pd.to_datetime(m.group(1)).date()
                end_date = pd.to_datetime(m.group(2)).date()
                return start_date, end_date
            except:
                continue

    # 如果找不到日期范围，尝试从单个日期推断
    single_date = _extract_date_from_filename(name)
    if single_date:
        date_obj = pd.to_datetime(single_date).date()
        return date_obj, date_obj

    return None, None


# --------------- 预处理（不改列名） ---------------

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        drop_cols = [c for c in ["Date", "Time", "Time Zone"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # 确保有日期列用于后续处理
    if "Datetime" in df.columns:
        df["date"] = pd.to_datetime(df["Datetime"], errors="coerce").dt.date

    for col in ["Net Sales", "Gross Sales", "Qty", "Discounts"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    if "Customer ID" not in df.columns:
        df["Customer ID"] = None

    return df


def preprocess_inventory(df: pd.DataFrame, filename: str = None) -> pd.DataFrame:
    df = _fix_header(df)
    required = [
        "Tax - GST (10%)", "Price", "Current Quantity Vie Market & Bar",
        "Default Unit Cost", "Categories"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None
    if filename:
        df["source_date"] = _extract_date_from_filename(filename)
    return df


def preprocess_members(df: pd.DataFrame) -> pd.DataFrame:
    return _fix_header(df)


# --------------- 表结构对齐 & 去重 & 写入 ---------------

def _table_exists(conn, table: str) -> bool:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    except Exception:
        return False


def _existing_columns(conn, table: str) -> list:
    try:
        cur = conn.execute(f"PRAGMA table_info('{table}')")
        return [row[1] for row in cur.fetchall()]
    except Exception:
        return []


def _add_missing_columns(conn, table: str, missing_cols: list, prefer_real: set):
    cur = conn.cursor()
    for col in missing_cols:
        coltype = "REAL" if col in prefer_real else "TEXT"
        cur.execute(f'''ALTER TABLE "{table}" ADD COLUMN "{col}" {coltype}''')
    conn.commit()


def _ensure_table_schema(conn, table: str, df: pd.DataFrame, prefer_real: set):
    if not _table_exists(conn, table):
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        return
    cols_now = set(_existing_columns(conn, table))
    incoming = list(df.columns)
    missing = [c for c in incoming if c not in cols_now]
    if missing:
        _add_missing_columns(conn, table, missing, prefer_real)


def _get_existing_date_ranges(conn, table: str):
    """获取数据库中已存在的日期范围"""
    try:
        if table == "transactions":
            query = "SELECT MIN(date), MAX(date) FROM transactions WHERE date IS NOT NULL"
            result = conn.execute(query).fetchone()
            if result and result[0] and result[1]:
                return pd.to_datetime(result[0]).date(), pd.to_datetime(result[1]).date()
    except Exception as e:
        pass
    return None, None


def _delete_overlapping_dates(conn, table: str, start_date, end_date, filename: str):
    """删除与新文件日期范围重叠的现有记录"""
    try:
        if table == "transactions" and start_date and end_date:
            # 删除与新文件日期范围完全重叠的现有记录
            delete_query = "DELETE FROM transactions WHERE date >= ? AND date <= ?"
            conn.execute(delete_query, (start_date, end_date))
            conn.commit()
            return True
    except Exception as e:
        pass
    return False


def _smart_deduplicate_transactions(df: pd.DataFrame, conn, filename: str):
    """智能去重逻辑：基于日期范围和Transaction ID"""
    if df.empty or "date" not in df.columns or "Transaction ID" not in df.columns:
        return df

    # 从文件名提取日期范围
    file_start_date, file_end_date = _extract_date_range_from_filename(filename)

    if not file_start_date or not file_end_date:
        # 如果无法从文件名提取日期范围，使用数据中的实际日期范围
        file_start_date = df["date"].min()
        file_end_date = df["date"].max()

    if file_start_date and file_end_date:
        # 删除该日期范围内的现有数据
        _delete_overlapping_dates(conn, "transactions", file_start_date, file_end_date, filename)

        # 返回所有数据（因为已经删除了重叠部分）
        return df
    else:
        # 回退到原来的去重逻辑
        return _deduplicate(df, ["Transaction ID", "date"], conn, "transactions")


def _deduplicate(df: pd.DataFrame, key_cols: list, conn, table: str) -> pd.DataFrame:
    """支持多列联合去重"""
    try:
        cols = [c for c in key_cols if c in df.columns]
        if not cols:
            return df
        query_cols = ", ".join([f'"{c}"' for c in cols])
        exist = pd.read_sql(f"SELECT {query_cols} FROM {table}", conn)
        if exist.empty:
            return df
        merged = df.merge(exist, on=cols, how="left", indicator=True)
        return merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    except Exception:
        return df


def _write_df(conn, df: pd.DataFrame, table: str, key_candidates: list, prefer_real: set, filename: str = None):
    _ensure_table_schema(conn, table, df, prefer_real)

    # === 针对 transactions 的智能处理 ===
    if table == "transactions":
        # 确保有日期列
        if "Datetime" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["Datetime"], errors="coerce").dt.date

        if "date" in df.columns and filename:
            # 使用智能去重逻辑
            df = _smart_deduplicate_transactions(df, conn, filename)
        else:
            # 回退到原来的去重逻辑
            df = _deduplicate(df, ["Transaction ID", "date"], conn, table)

    # === 针对 inventory ===
    elif table == "inventory" and "source_date" in df.columns:
        dates = df["source_date"].dropna().unique().tolist()
        if dates:
            for d in dates:
                conn.execute(f'DELETE FROM "{table}" WHERE source_date=?', (d,))
            conn.commit()
        # ✅ 改为 SKU + source_date 联合去重
        df = _deduplicate(df, ["SKU", "source_date"], conn, table)

    # === 默认情况（members等） ===
    else:
        key_col = next((k for k in key_candidates if k in df.columns), None)
        if key_col:
            df = _deduplicate(df, [key_col], conn, table)

    if not df.empty:
        df.to_sql(table, conn, if_exists="append", index=False)
        return len(df)
    return 0


# --------------- 索引 ---------------
def ensure_indexes():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU)')
    conn.commit()


# --------------- 从 Google Drive 导入 ---------------
def ingest_from_drive_all():
    conn = get_db()
    drive = get_drive()
    files = drive.ListFile({'q': f"'{FOLDER_ID}' in parents and trashed=false"}).GetList()
    if not files:
        return

    # 按文件名排序，确保较新的文件在后面处理（如果需要覆盖逻辑）
    files.sort(key=lambda x: x["title"])

    seen = set()
    total_imported = 0

    for f in files:
        name = f["title"]
        if name in seen:
            continue
        seen.add(name)

        local = os.path.join(tempfile.gettempdir(), name)
        f.GetContentFile(local)

        if name.lower().endswith(".csv"):
            df = pd.read_csv(local)
            df = _fix_header(df)
        elif name.lower().endswith(".xlsx"):
            header_row = 1 if "catalogue" in name.lower() else 0
            df = pd.read_excel(local, header=header_row)
            df = _fix_header(df)
        else:
            continue

        imported_count = 0
        if "items" in name.lower():
            df = preprocess_transactions(df)
            imported_count = _write_df(conn, df, "transactions",
                                       key_candidates=["Transaction ID", "Item", "Price", "Modifiers Applied"],
                                       prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"},
                                       filename=name)

        elif "catalogue" in name.lower():
            df = preprocess_inventory(df, filename=name)
            imported_count = _write_df(conn, df, "inventory",
                                       key_candidates=["SKU"], prefer_real=set(),
                                       filename=name)

        elif "member" in name.lower():
            df = preprocess_members(df)
            imported_count = _write_df(conn, df, "members",
                                       key_candidates=["Square Customer ID", "Reference ID"],
                                       prefer_real=set(), filename=name)

        total_imported += imported_count

        try:
            os.remove(local)
        except Exception:
            pass

    ensure_indexes()
    return True


@st.cache_data(show_spinner=False)
def init_db_from_drive_once():
    try:
        ingest_from_drive_all()
    except Exception as e:
        pass
    return True


# --------------- 手动导入（Sidebar 上传） ---------------
def ingest_csv(uploaded_file):
    conn = get_db()
    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.csv"

    df = pd.read_csv(uploaded_file)
    df = _fix_header(df)

    imported_count = 0
    try:
        if "Net Sales" in df.columns and "Gross Sales" in df.columns:
            df = preprocess_transactions(df)
            imported_count = _write_df(conn, df, "transactions",
                                       key_candidates=["Transaction ID"],
                                       prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"},
                                       filename=filename)

        elif "SKU" in df.columns or "Stock on Hand" in df.columns or "Categories" in df.columns:
            df = preprocess_inventory(df, filename=filename)
            imported_count = _write_df(conn, df, "inventory",
                                       key_candidates=["SKU"], prefer_real=set(),
                                       filename=filename)

        elif "Square Customer ID" in df.columns or "First Name" in df.columns or "member" in filename.lower():
            df = preprocess_members(df)
            imported_count = _write_df(conn, df, "members",
                                       key_candidates=["Square Customer ID", "Reference ID"],
                                       prefer_real=set(), filename=filename)

        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            if hasattr(uploaded_file, "getbuffer"):
                f_local.write(uploaded_file.getbuffer())
            else:
                f_local.write(uploaded_file.read())
        upload_file_to_drive(tmp_path, filename)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    ensure_indexes()
    return True


def ingest_excel(uploaded_file):
    conn = get_db()
    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.xlsx"

    data = uploaded_file.read()
    xls = pd.ExcelFile(BytesIO(data))

    total_imported = 0
    try:
        for sheet in xls.sheet_names:
            header_row = 1 if ("catalogue" in filename.lower()) else 0
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
            df = _fix_header(df)

            imported_count = 0
            if "Net Sales" in df.columns and "Gross Sales" in df.columns:
                df = preprocess_transactions(df)
                imported_count = _write_df(conn, df, "transactions",
                                           key_candidates=["Transaction ID"],
                                           prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"},
                                           filename=filename)

            elif "SKU" in df.columns or "Stock on Hand" in df.columns or "Categories" in df.columns:
                df = preprocess_inventory(df, filename=filename)
                imported_count = _write_df(conn, df, "inventory",
                                           key_candidates=["SKU"], prefer_real=set(),
                                           filename=filename)

            elif "Square Customer ID" in df.columns or "First Name" in df.columns or "member" in filename.lower():
                df = preprocess_members(df)
                imported_count = _write_df(conn, df, "members",
                                           key_candidates=["Square Customer ID", "Reference ID"],
                                           prefer_real=set(), filename=filename)

            total_imported += imported_count

        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(data)
        upload_file_to_drive(tmp_path, filename)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    ensure_indexes()
    return True


__all__ = [
    "ingest_csv",
    "ingest_excel",
    "ingest_from_drive_all",
    "get_drive",
    "upload_file_to_drive",
    "download_file_from_drive",
    "init_db_from_drive_once",
]