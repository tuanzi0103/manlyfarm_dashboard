# services/db.py
from functools import lru_cache

def _get_secret(name: str, default: str = None):
    # 优先从 Streamlit secrets 里取，其次从系统环境变量取
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    import os
    return os.getenv(name, default)

@lru_cache()
def get_db():
    from pymongo import MongoClient
    mongo_uri = _get_secret("MONGODB_URI")             # 必填
    db_name   = _get_secret("MONGODB_DB", "manly_farm")# 可改成你的库名

    if not mongo_uri:
        raise RuntimeError("Missing MONGODB_URI in secrets or env")

    client = MongoClient(mongo_uri)
    return client[db_name]
