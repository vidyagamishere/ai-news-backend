#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()

from db_service import get_database_service

db = get_database_service()
result = db.execute_query('SELECT * FROM publishers_master LIMIT 1', fetch_one=True)
if result:
    print("Publishers_master columns:")
    for col in result.keys():
        print(f"  - {col}")
else:
    print("No data in publishers_master")
