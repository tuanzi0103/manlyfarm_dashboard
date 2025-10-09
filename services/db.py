# services/db.py
import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "manlyfarm.db")

def get_db():
    """每次都创建新的数据库连接，避免缓存问题"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn