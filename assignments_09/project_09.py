# https://youtu.be/FHh1s5ddi7A
import requests
from dotenv import load_dotenv
import os
from supabase import create_client
from datetime import datetime

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.3861,     # Location: Mountain View, CA
    "longitude": -122.0839,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params = params)
response.raise_for_status()
data = response.json()
daily = data["daily"]

records = [{
    "date": daily["time"][i],
    "temperature_2m_max": daily["temperature_2m_max"][i],
    "temperature_2m_min": daily["temperature_2m_min"][i],
    "precipitation_sum": daily["precipitation_sum"][i],
     "wind_speed_10m_max": daily[ "wind_speed_10m_max"][i]
    } 
    for i in range(len(daily["time"]))
]

print("The first record: ", records[0])
print(f"The last {len(records)} record: {records[len(records)-1]}" )

# For a full year I expect 365 records, and I get exact that amount. If the numbers differ, it means that either some records were not downloaded correctly 
# or they were absent from original data.

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print("'Warning: could not load environment variables from .env'")

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

if not SUPABASE_KEY or not SUPABASE_URL:
    raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variable."
        )
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sup_response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
print(f"{len(sup_response.data)} rows were inserted in the table")

sup_response = supabase.table("weather_raw").select("*", count="exact").execute()
print(f"{sup_response.count} rows were selected")
# Second run went smoothly without errors and the number of rows didn't change, so no dublicate data were inserted. The code is idempotent.
# sup_response_latest_date = 
print(f"The earliest record: {sup_response.data[0]}")
print(f"The latest record: {sup_response.data[-1]}")
target_date = "2023-07-04"
row_selected = next(
    (row for row in sup_response.data if row["date"] == target_date), 
    None)
if row_selected is None:
    
    for row in sup_response.data:
        row_selected = min(sup_response.data, key=lambda row: abs(datetime.strptime(row["date"], "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")))
print(f"Record for date {target_date}: {row_selected}")



