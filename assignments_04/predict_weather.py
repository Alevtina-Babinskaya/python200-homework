import joblib
import pandas as pd
import json

model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)
print("Location (latitude and longitude):", metadata["location"])
print("Test AUC:", metadata["test_auc"])
print("Features:", metadata['features'])

import pandas as pd

new_days = pd.DataFrame([
    # clearly good day
    {
        "temperature_2m_max": 20,
        "temperature_2m_min": 10,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 10,
    },
    # another good day
    {
        "temperature_2m_max": 15,
        "temperature_2m_min": 5,
        "precipitation_sum": 0.5,
        "wind_speed_10m_max": 15,
    },
    # clearly bad: too hot
    {
        "temperature_2m_max": 35,
        "temperature_2m_min": 20,
        "precipitation_sum": 2,
        "wind_speed_10m_max": 10,
    },
    # clearly bad: rain and strong wind
    {
        "temperature_2m_max": 18,
        "temperature_2m_min": 8,
        "precipitation_sum": 10,
        "wind_speed_10m_max": 35,
    },
    # borderline case
    {
        "temperature_2m_max": 22,
        "temperature_2m_min": 2,
        "precipitation_sum": 2.5,
        "wind_speed_10m_max": 24,
    },
])
print(new_days)
predictions = model.predict(new_days)
probs = model.predict_proba(new_days)[:, 1]
for i, (row, pred, prob) in enumerate(zip(new_days.values, predictions, probs)):
    label = "good" if pred == 1 else "skip"
    print(f"Day {i+1}:")
    print(f"Features: {row}")
    print(f"Prediction: {label} ({prob:.4f} probability)")



# In my border case the probability was 0.0024. Despite the obviously positive values for classifying this day as good for running,
# the model confidently classified it as 0. I would handle a day where the model says 0.52 according to the goals of the prediction. 
# In case of good for running days prediction I would classify it as "skip"

# If predict_weather.py is run before train_weather_classifier.py, the saved model file (weather_classifier.pkl) will not exist, causing a FileNotFoundError.
# To make the error more helpful, the prediction script should check whether the model file exists and display a message telling to run the training script first.

# In a production system, predict_weather.py should automatically fetch the latest weather forecast data from an API, convert the response into
# a DataFrame with the correct feature names, and run the saved model on the new data. The prediction pipeline should be scheduled to run daily 
# so that users receive updated weather recommendations.