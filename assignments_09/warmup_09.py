# -------- Supabase Connection --------------
# ---- Connection Question 1 -----
# To connect to the project, supabase-py needs API key and link to the project. This two pieces are placed in .env file.  
# They can be found in the Project Settings section of the dashboard. API key is under API Keys subsection and the link is in Data API -> docs section.
# They should never be hardcoded in a Python script because in that case they would committed to GitHub, become public and be scraped by bots. 
# After that anyone can access the database and use data in it.

# ----- Connection Question 2 ------
from dotenv import load_dotenv
import os
from datetime import date
from supabase import create_client
def get_client():
    if load_dotenv():
        print('Successfully loaded environment variables from .env')
    else:
        print("'Warning: could not load environment variables from .env'")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_KEY or not SUPABASE_URL:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variable."
        )
        
    else:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    
# ----- Connection Question 3 ------
# Row Level Security (RLS) is a database security feature that controls which rows a particular user is allowed to access.
# We disabled RLS for this course because we working on educational project with one user, so row-level access control is not necessary. 
# When RLS is enabled without properly configured policies, it creates problems during development, raising access errors when trying to insert or update rows
# I want to keep it enabled when data is used by multiple users and everyone has its own role and permissions.

# ------- supabase-py CRUD -----------
# ---- CRUD Question 1 ------
def insert_test_record(supabase):
    today = date.today().isoformat()
    row = { 
        "date":               today,
        "temperature_2m_max": 26,
        "temperature_2m_min": 14,
        "precipitation_sum":  0.04,
        "wind_speed_10m_max": 6.0,
    }
    supabase.table("weather_raw").insert(row).execute()
supabase = get_client()
insert_test_record(supabase)
# If I ran the function twice, I would get a "duplicate key value" error because a record with the same key already exists.
# I would change the '.insert' command to '.upsert'. In this case, if a row with the given key already exists, it will be updated.

# ----- CRUD Question 2 ------
def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").lte("date", end).gte("date", start).execute()
    return response.data
date_range = get_records_by_date_range(supabase, "2026-08-26", "2026-09-01")
print(date_range)

# ------ CRUD Question 3 -------
# '.insert' command adds rows to the table. If rows with the same unique keys already exist in the table, Supabase-py raises an error.
# '.upsert' command either updates an existing row or adds a new row to the table.
# I would use '.insert' to insert rows if I don't want them to be updated accidentally.
# In other cases, I would use '.upsert' because it allows me to run the code many times without errors.
def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"{len(response.data)} rows were affected")

# ------------- Idempotency ---------------
# ------ Idempotency Question 1 -----
# A data pipeline is a set of operations that cleans, transforms, and enriches data. 
# Some pipelines are designed to load data automatically on a schedule.
# Idempotency is important because if something goes wrong, the pipeline should be able to run again 
# and produce the expected result without creating duplicates or damaging the data.
# For example, if a pipeline loads 1,000 rows but crashes after loading 600, a non-idempotent pipeline 
# may load those 600 rows again when it is restarted. This would create duplicate rows and 
# could affect calculations and analysis based on the data.



