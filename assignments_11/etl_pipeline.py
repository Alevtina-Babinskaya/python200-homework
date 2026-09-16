# Link to the video presentation https://youtu.be/3nsr6R55FXk
from prefect import flow, task
from prefect.logging import get_run_logger
import os
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import requests
import joblib
import pandas as pd

load_dotenv()
if load_dotenv:
    print("Environment variables were successfully loaded")
else:
    print("Failed to load environment variables")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_KEY or not SUPABASE_URL:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variable.")
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("assignments_11/models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)
LATITUDE = 37.3861 # Location: Mountain View, CA
LONGITUDE = -122.0839

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,     
        "longitude": LONGITUDE,
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
    daily = response.json()["daily"]
    records = [{
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i]
    } for i in range(len(daily["time"]))]
    print(f"{len(records)} records were extracted from Open-Meteo")
    return records

@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"{len(response.data)} records were upserted in raw weather table")

@task
def transform(raw_records: list) -> list:
    records_completed = {r["date"] for r in supabase.table("weather_enriched").select("date").execute().data}
    to_complete = [r for r in raw_records if r["date"] not in records_completed]
    print(f"{len(to_complete)} of {len(raw_records)} records to transform")
    if not to_complete:
        print("All records are already enriched")
        return []

    clf = joblib.load("assignments_11/models/weather_classifier.pkl")
    df = pd.DataFrame(to_complete)
    X = df[FEATURES]
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    print(f"ML classification complete. Good days: {int(preds.sum())} / {len(preds)}")
    enrichment_records = [
        {
        "date": to_complete[i]["date"],
        "good_for_running": bool(preds[i]),
        "confidence": round(float(probs[i]), 10),
        "llm_summary": None
        } for i in range(len(to_complete))]
    for i, record in enumerate(enrichment_records):
        raw_row = to_complete[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
            )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message}],
                max_tokens=100
            )
            record["llm_summary"] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."
        if (i +1)%50 == 0:
            print(f"{i+1} out of {len(enrichment_records)} records completed")
    print(f"Transform complete: {len(enrichment_records)} records enriched")
    return enrichment_records
    
@task(retries=2, retry_delay_seconds=5)
def load_enriched(enriched_records: list) -> None:
    if not enriched_records:
        print("Warning: there is nothing to load.")
        return
    response = supabase.table("weather_enriched").upsert(enriched_records, on_conflict="date").execute()
    print(f"{len(response.data)} records successfully uploaded to the table")

@flow(log_prints=True)
def etl_pipeline():
    raw_records = extract()
    load_raw(raw_records)
    enriched_records = transform(raw_records)
    load_enriched(enriched_records)
    print("Pipeline complete.")

if __name__ == "__main__":
    etl_pipeline()
        

