import os
import pandas as pd
from io import BytesIO
import streamlit as st
from services.db import get_db

# === Google Drive 相关 ===
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
from services.db import DB_PATH

# ✅ 启动时仅在数据库不存在时执行一次导入
def init_db_from_drive_once():
    if not os.path.exists(DB_PATH) or os.stat(DB_PATH).st_size == 0:
        try:
            ingest_from_drive_all()
            print("✅ Database initialized from Google Drive")
        except Exception as e:
            print(f"⚠️ Failed to initialize DB from Google Drive: {e}")


FOLDER_ID = "1lZGE0DkgKyox1HbBzuhZ-oypDF478jBj"

# ✅ 全局缓存 drive 实例
_drive_instance = None

def get_drive():
    global _drive_instance
    if _drive_instance is not None:
        return _drive_instance

    gauth = GoogleAuth()

    # ✅ 尝试加载本地保存的 token
    token_path = "credentials.json"
    if os.path.exists(token_path):
        gauth.LoadCredentialsFile(token_path)

    if gauth.credentials is None:
        # 第一次认证：弹浏览器 → 保存 token
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # token 过期 → 刷新
        gauth.Refresh()
    else:
        # 直接授权
        gauth.Authorize()

    # ✅ 保存到本地，后续不会再弹
    gauth.SaveCredentialsFile(token_path)

    _drive_instance = GoogleDrive(gauth)
    return _drive_instance


def upload_file_to_drive(local_path, title):
    drive = get_drive()
    f = drive.CreateFile({'title': title, 'parents': [{'id': FOLDER_ID}]})
    f.SetContentFile(local_path)
    f.Upload()
    return f['id']

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

# --------------- 预处理（不改列名） ---------------

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)

    # 生成统一 Datetime，并移除 Date/Time/Time Zone（避免列冲突）
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        drop_cols = [c for c in ["Date", "Time", "Time Zone"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # 仅清洗数值，不改列名
    for col in ["Net Sales", "Gross Sales", "Qty", "Discounts"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    return df

def preprocess_inventory(df: pd.DataFrame) -> pd.DataFrame:
    return _fix_header(df)

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
        return [row[1] for row in cur.fetchall()]  # [cid, name, type, notnull, dflt, pk]
    except Exception:
        return []

def _add_missing_columns(conn, table: str, missing_cols: list, prefer_real: set):
    cur = conn.cursor()
    for col in missing_cols:
        coltype = "REAL" if col in prefer_real else "TEXT"
        cur.execute(f'''ALTER TABLE "{table}" ADD COLUMN "{col}" {coltype}''')
    conn.commit()

def _ensure_table_schema(conn, table: str, df: pd.DataFrame, prefer_real: set):
    """
    确保数据库中的 {table} 拥有 df 的所有列：
    - 表不存在：用 df 的列直接建表（空表），再 append 数据；
    - 表已存在：缺啥列就 ALTER TABLE ADD COLUMN（不丢旧数据）。
    """
    if not _table_exists(conn, table):
        # 用 df 的结构创建空表
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        return

    cols_now = set(_existing_columns(conn, table))
    incoming = list(df.columns)
    missing = [c for c in incoming if c not in cols_now]
    if missing:
        _add_missing_columns(conn, table, missing, prefer_real)

def _deduplicate(df: pd.DataFrame, key_col: str, conn, table: str) -> pd.DataFrame:
    """按 key 去重（仅过滤已存在的）"""
    if key_col not in df.columns:
        return df
    try:
        exist = pd.read_sql(f'''SELECT "{key_col}" FROM "{table}"''', conn)
        existed_set = set(exist[key_col].dropna().astype(str).unique())
        mask = ~df[key_col].astype(str).isin(existed_set)
        return df[mask]
    except Exception:
        return df  # 表可能刚创建，还没有数据

def _write_df(conn, df: pd.DataFrame, table: str, key_candidates: list, prefer_real: set):
    # 1) 表结构对齐（自动补列）
    _ensure_table_schema(conn, table, df, prefer_real)

    # 2) 找到可用主键做去重
    key_col = next((k for k in key_candidates if k in df.columns), None)
    if key_col:
        df = _deduplicate(df, key_col, conn, table)

    # 3) 只写入当前 df 的列（列名对齐后不会再报错）
    if not df.empty:
        df.to_sql(table, conn, if_exists="append", index=False)

# --------------- 索引（非唯一，避免失败） ---------------

def ensure_indexes():
    conn = get_db()
    cur = conn.cursor()
    # transactions
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID])')
    # members
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID])')
    # inventory
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU)')
    conn.commit()

# --------------- CSV/Excel 导入（保留原逻辑） ---------------

import os
import tempfile
import pandas as pd
import streamlit as st
from io import BytesIO
from services.db import get_db
from services.ingestion import upload_file_to_drive, _fix_header, preprocess_transactions, preprocess_inventory, preprocess_members

# === CSV 导入 ===
def ingest_csv(uploaded_file):
    conn = get_db()
    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.csv"
    st.sidebar.info(f"📂 Importing {filename}")

    df = pd.read_csv(uploaded_file)
    df = _fix_header(df)

    try:
        if "Net Sales" in df.columns and "Gross Sales" in df.columns:
            df = preprocess_transactions(df)
            df.to_sql("transactions", conn, if_exists="append", index=False)
            st.sidebar.success(f"✅ Imported {len(df)} rows (transactions)")

        elif "SKU" in df.columns or "Stock on Hand" in df.columns:
            df = preprocess_inventory(df)
            df.to_sql("inventory", conn, if_exists="append", index=False)
            st.sidebar.success(f"✅ Imported {len(df)} rows (inventory)")

        elif "Square Customer ID" in df.columns or "First Name" in df.columns:
            df = preprocess_members(df)
            df.to_sql("members", conn, if_exists="append", index=False)
            st.sidebar.success(f"✅ Imported {len(df)} rows (members)")

        else:
            st.sidebar.warning(f"⚠️ Skipped {filename}, schema not recognized")

        # ✅ 保存上传的文件到临时目录 → 再上传到 Google Drive
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(uploaded_file.getbuffer())
        upload_file_to_drive(tmp_path, filename)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return True


# === Excel 导入 ===
def ingest_excel(uploaded_file):
    conn = get_db()
    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.xlsx"
    st.sidebar.info(f"📂 Importing {filename}")

    data = uploaded_file.read()
    xls = pd.ExcelFile(BytesIO(data))

    try:
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            df = _fix_header(df)

            if "Net Sales" in df.columns and "Gross Sales" in df.columns:
                df = preprocess_transactions(df)
                df.to_sql("transactions", conn, if_exists="append", index=False)

            elif "SKU" in df.columns or "Stock on Hand" in df.columns:
                df = preprocess_inventory(df)
                df.to_sql("inventory", conn, if_exists="append", index=False)

            elif "Square Customer ID" in df.columns or "First Name" in df.columns:
                df = preprocess_members(df)
                df.to_sql("members", conn, if_exists="append", index=False)

        st.sidebar.success(f"✅ {filename} imported")

        # ✅ 保存上传的文件到临时目录 → 再上传到 Google Drive
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(data)
        upload_file_to_drive(tmp_path, filename)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return True


# --------------- 启动时从 Drive 全量导入 ---------------

def ingest_from_drive_all():
    conn = get_db()
    drive = get_drive()
    files = drive.ListFile({'q': f"'{FOLDER_ID}' in parents and trashed=false"}).GetList()

    if not files:
        st.warning("⚠️ Google Drive folder is empty.")
        return

    for f in files:
        name = f["title"]
        local = os.path.join("/tmp", name)
        f.GetContentFile(local)

        if name.lower().endswith(".csv"):
            df = pd.read_csv(local)
            df = _fix_header(df)

            if "Net Sales" in df.columns and "Gross Sales" in df.columns:
                df = preprocess_transactions(df)
                _write_df(
                    conn, df, "transactions",
                    key_candidates=["Transaction ID"],
                    prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"}
                )

            elif "SKU" in df.columns or "Variation Name" in df.columns:
                df = preprocess_inventory(df)
                _write_df(conn, df, "inventory", key_candidates=["SKU"], prefer_real=set())

            elif "Square Customer ID" in df.columns or "Reference ID" in df.columns or "First Name" in df.columns:
                df = preprocess_members(df)
                _write_df(conn, df, "members",
                          key_candidates=["Square Customer ID", "Reference ID"],
                          prefer_real=set())

        elif name.lower().endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(local)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=0)
                df = _fix_header(df)

                if "Net Sales" in df.columns and "Gross Sales" in df.columns:
                    df = preprocess_transactions(df)
                    _write_df(
                        conn, df, "transactions",
                        key_candidates=["Transaction ID"],
                        prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"}
                    )

                elif "SKU" in df.columns or "Variation Name" in df.columns:
                    df = preprocess_inventory(df)
                    _write_df(conn, df, "inventory", key_candidates=["SKU"], prefer_real=set())

                elif "Square Customer ID" in df.columns or "Reference ID" in df.columns or "First Name" in df.columns:
                    df = preprocess_members(df)
                    _write_df(conn, df, "members",
                              key_candidates=["Square Customer ID", "Reference ID"],
                              prefer_real=set())

    ensure_indexes()
    print("✅ Google Drive ingestion finished (schema-evolved, deduped & indexed)")

# --------------- 导出 ---------------

__all__ = [
    "ingest_csv",
    "ingest_excel",
    "ingest_from_drive_all",
    "get_drive",
    "upload_file_to_drive",
    "download_file_from_drive",
]
