import os
import streamlit as st
from pymongo import MongoClient

def get_db():
    mongo_uri = (
        st.secrets.get("MONGODB_URI")
        or os.environ.get("MONGODB_URI")
    )
    db_name = (
        st.secrets.get("MONGODB_DB")
        or os.environ.get("MONGODB_DB")
        or "manly_farm"
    )

    if not mongo_uri:
        raise RuntimeError("Missing MONGODB_URI in secrets or env")

    client = MongoClient(mongo_uri)
    return client[db_name]
