import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import numpy as np
import json
import sklearn
import sys


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
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

def labeling(row):
    return int(
        7 <= row["temperature_2m_max"] <= 26 
        and row["temperature_2m_min"] >= 0
        and row["precipitation_sum"] < 3.0
        and row["wind_speed_10m_max"] < 25 # 25 km/h is a strong wind already
    )
df["good_for_running"] = df.apply(labeling, axis = 1)
print(df["good_for_running"].value_counts())
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[FEATURES]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
params = {"model__C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    estimator = pipe,
    param_grid = params,
    cv = 5,
    scoring="roc_auc",
    n_jobs = -1
)
grid_search.fit(X_train, y_train)
print(f"Best C: {grid_search.best_params_['model__C']:.3f}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
best_pipe = grid_search.best_estimator_
y_pred = best_pipe.predict(X_test)
y_probs = best_pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC = {grid_search.best_score_:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.close()

threshold_df = pd.DataFrame({
    "threshold": thresholds,
    "fpr":       fpr,
    "tpr":       tpr,
}).round(3)

# Find the threshold closest to a target TPR of 0.90
target_fpr = 0.1
idx = np.argmin(np.abs(threshold_df["fpr"] - target_fpr))
print(f"Fore {target_fpr} the threshold is {threshold_df.iloc[idx]}")

# The test AUC is 0.96, which indicates excellent performance and is about what I expected because the labels were created from the weather features.

# Precision (0.91) is slightly lower than recall (0.95), so false positives are slightly more common than false negatives.
# This means the app sometimes recommends running on days that are not actually good.
# I would rather the app under-recommend running than over-recommend it because recommending a bad running day could lead to an unpleasant or unsafe experience.

# I would not use the default threshold of 0.5.
# I would increase it to about 0.70 because this reduces the false positive rate to about 6.7%.
# Although the true positive rate decreases to 88.4%, I think this is a better trade-off for a running recommendation app.

joblib.dump(best_pipe, "models/weather_classifier.pkl")
print("Model saved")
metadata = {
    "python_version":   sys.version,
    "sklearn_version":  sklearn.__version__,
    "features":         FEATURES,
    "best_params":      grid_search.best_params_,
    "test_auc":         roc_auc_score(y_test, y_probs),
    "location":         {
        "latitude": 37.3861,   
        "longitude": -122.0839},
    "label_thresholds": {
        "temperature_2m_max": "7–26°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 25 km/h",
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved")