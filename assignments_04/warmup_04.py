import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ROC and AUC
# ROC Question 1
lrm = LogisticRegression(max_iter=1000, random_state=42)
lrm.fit(X_train, y_train)
y_probs_lrm = lrm.predict_proba(X_test)[:, 1]
auc_lrm = roc_auc_score(y_test, y_probs_lrm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]
auc_knn = roc_auc_score(y_test, y_probs_knn)
print("Logistic Regression AUC: ", auc_lrm)
print("KNN AUC: ", auc_knn)
# KNN model has higher AUC (0.94) than Logistic Regression (0.706). KNN better separates the two classes.

# ROC Question 2
fpr_lrm, tpr_lrm, thresholds_lrm = roc_curve(y_test, y_probs_lrm)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_probs_knn)
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lrm, tpr=tpr_lrm).plot(ax=ax, name=f"Logistic Regression (AUC={auc_lrm:.2f})")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn).plot(ax=ax, name=f"KNN Model (AUC={auc_knn:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("Logistic Regression and KNN Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.close()
# The plot shows that KNN model has fewer FPR at TPR = 0.80 - 0.08 for KNN vs 0.6 for LR. If I needed to catch 80% of positives, KNN model would produce fewer false alarms (less than 10%) 

# ROC Question 3
best_f1 = 0
for i, threshold in enumerate(thresholds_lrm):
    y_pred = (y_probs_lrm >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tpr = tpr_lrm[i]
        best_fpr = fpr_lrm[i]
print(f"Optimal threshold: {best_threshold:.2f}")
print("TPR:", best_tpr)
print("FPR:", best_fpr)
print(f"F1: {best_f1:.2f}")
# The optimal threshold is equal to 0.28 which is lower than the default 0.5.  
# In a real application, I would choose a threshold lower than 0.5 when missing a true positive is more costly than having some false positives.

# GridSearchCV
# GridSearch Question 1
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs = -1
)
grid_search.fit(X_train, y_train)
print(f"Best C: {grid_search.best_params_['model__C']:.3f}")
print(f"Best CV AUC score: {grid_search.best_score_:.3f}")
best_pipe = grid_search.best_estimator_
y_probs = best_pipe.predict_proba(X_test)[:,1]
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")
# The grid search did not select the default value; it chose C = 100. However, the test AUC remained 0.706, the same as with the default C = 1.0

# GridSearch Question 2
pipe_dt = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier())
])
param_grid_dt = {"model__max_depth": [2, 3, 5, 8, None]}
grid_search_dt = GridSearchCV(
    estimator = pipe_dt,
    param_grid = param_grid_dt,
    cv=5,
    scoring="roc_auc",
    n_jobs = -1
)
grid_search_dt.fit(X_train, y_train)
print(f"Best max_depth: {grid_search_dt.best_params_['model__max_depth']:.3f}")
print(f"Best CV AUC score: {grid_search_dt.best_score_:.3f}")
best_pipe_dt = grid_search_dt.best_estimator_
y_probs_dt = best_pipe_dt.predict_proba(X_test)[:, 1]
print(f"Test AUC Desicion Tree: {roc_auc_score(y_test, y_probs_dt):.3f}")
# The Decision Tree model achieved a higher test AUC than Logistic Regression.
# However, KNN achieved the highest test AUC overall, so I would choose KNN for further development.
# AUC is not the only factor to consider. I would also look at precision, recall, F1 score, model stability.

# GridSearch Question 3
results_lr = pd.DataFrame(grid_search.cv_results_)
print(results_lr[["param_model__C", "mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False).to_string(index=False))
results_dt = pd.DataFrame(grid_search_dt.cv_results_)
print(results_dt[["param_model__max_depth", "mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False).to_string(index=False))
# If I had to choose between models with similar mean scores, I would pick the one with the lower standard deviation
# because it is more stable across different cross-validation folds.

# joblib
# joblib Question 1
joblib.dump(best_pipe, "models/warmup_model.pkl")
loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
# If I saved only the logistic regression model (without the scaler), the results on prediction would be different, 
# because model knows nothing about how training data were scaled.

# joblib Question 2
# --- Simulated prediction script ---
loaded_model = joblib.load("models/warmup_model.pkl")
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])
new_pred = loaded_model.predict(new_samples)
new_probs = loaded_model.predict_proba(new_samples)[:, 1]
print(new_pred)
print(new_probs)
# I expected the model to predict class 0 for the third sample because all of its feature values are zero.
# However, logistic regression also includes an intercept, so even with all-zero features
# the predicted probability can result in class 1.