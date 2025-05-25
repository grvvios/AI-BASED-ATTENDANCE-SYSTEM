from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["ai_attendance"]
collection = db["attendance"]

def store_attendance(name):
    current_time = datetime.now()
    existing = collection.find_one({"name": name, "date": current_time.strftime("%Y-%m-%d")})
    
    if not existing:
        attendance_data = {
            "name": name,
            "time": current_time.strftime("%H:%M:%S"),
            "date": current_time.strftime("%Y-%m-%d")
        }
        collection.insert_one(attendance_data)
