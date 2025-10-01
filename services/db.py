# services/db.py
import sqlite3
from functools import lru_cache

DB_PATH = "manlyfarm.db"

@lru_cache()
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
