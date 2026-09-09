# The link for video presentation: https://youtu.be/eMAQEnLXCWk
import json
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import joblib
from openai import OpenAI
import re
with open("assignments_10/models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
response = supabase.table("weather_raw").select("*").execute()
raw_rows =response.data
enriched_response = supabase.table("weather_enriched").select("date").execute()
already_done = {row["date"] for row in enriched_response.data}
to_classify = [row for row in raw_rows if row["date"] not in already_done]

print(f"There {len(raw_rows)} records in the raw weather table, {len(already_done)} are already processed, {len(to_classify)} remain to classify")

clf = joblib.load("assignments_10/models/weather_classifier.pkl")
df = pd.DataFrame(to_classify)
X = df[FEATURES]
preds = clf.predict(X)
probs = clf.predict_proba(X)[:, 1]
# confidence = np.max(probs, axis=1)
print(f"Good days predicted: {preds.sum()} / {len(preds)}")
print(f"Confidence range: {probs.min():.2f} – {probs.max():.2f}")
enrichment_data = []
for i, row in enumerate(to_classify):
    enrichment_data.append({"date": row['date'],
                            "good_for_running": bool(preds[i]),
                            "confidence": float(probs[i]),
                            })
    
SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app."
    "You will receive weather conditions for a single day and a machine learning prediction about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions."
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

def validate_summary(text):
    text = text.strip()
    if not text:
        return None
    sentences = re.split(r'(?<!\d)\.(?!\d)', text) # splits text into sentenses by '.' ignoring them in numbers 
    if len(sentences) > 2:
        return None
    return text

for i, row in enumerate(enrichment_data):
    raw_row = next(r for r in to_classify if r["date"] == row["date"])
    try:
        llm_response = llm.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(
                    raw_row,
                    row["good_for_running"],
                    row["confidence"],
                    ),
                },
            ],
            max_tokens=100,
        )
        summary = validate_summary(llm_response.choices[0].message.content.strip())
        
    except Exception as e:
        print(f"  API error on {row['date']}: {e}")
        summary = "Recommendation unavailable."
    row["llm_summary"] = summary
    if (i + 1) % 50 == 0:
        print(f"  Enriched {i + 1} / {len(enrichment_data)} records...")
response = supabase.table("weather_enriched").upsert(enrichment_data, on_conflict="date").execute()
print(f"Upserted {len(response.data)} rows into weather_enriched")

verify_response = supabase.table("weather_enriched").select("*").execute()
print(f"There are {len(verify_response.data)} records in the enriched table")
print("\nFive sample rows:")
for row in verify_response.data[:5]:
    print({
        "date": row["date"],
        "good_for_running": row["good_for_running"],
        "confidence": row["confidence"],
        "llm_summary": row["llm_summary"]
    })

good_days = sum(row["good_for_running"] for row in verify_response.data)
print(f"\nNumber of days classified as good for running: {good_days}")
# Most of the summaries include accurate data and clear prediction messages. 
# "With a high of 13.7°C, low of 8.8°C, and significant rainfall of 16.2 mm along with strong winds, today is not ideal for running." 
# matches the prompt instructions in a best way. It's practical, direct and specific.
# "With rain and strong winds expected, today is not ideal for running." - this kind of summary lacks specific data.
# The weaker summary may be the result of extreme weather conditions in the raw record, 
# causing the LLM to focus on the general recommendation rather than providing all the specific data.

# Reflection
# I would not necessarily expect the classifier's predictions to be accurate for a different city. 
# The classifier was trained on Charlotte, NC weather data, so it learned patterns from that climate. 
# If the new city's weather is significantly different, the model may not generalize well.

# The LLM can potentially override the classifier if the prompt allows it. 
# In this pipeline, however, it is mainly additive: it uses the classifier's prediction and 
# weather features to generate a recommendation. The LLM could classify the data itself, 
# but it would be less reliable for this type of structured classification and would increase cost and latency.

# For 50,000 records, my main concerns would be latency and cost. 
# If each record requires a separate LLM call, processing them sequentially could take a very long time. 
# I would use asynchronous or concurrent API calls to process multiple records at the same time.

