import sqlite3
from services.db import DB_PATH

def init_db():
    """初始化 SQLite 数据库，不预建表，等 ingestion 自动创建"""
    conn = sqlite3.connect(DB_PATH)
    conn.close()
    print(f"✅ Database ready at {DB_PATH}")

if __name__ == "__main__":
    init_db()
