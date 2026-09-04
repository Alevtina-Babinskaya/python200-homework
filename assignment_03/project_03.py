import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay

)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

# Task 1
sns.boxplot(data = df, x = "spam_label", y = "word_freq_free")
plt.xlabel("Email type")
plt.ylabel("Word frequency free")
plt.savefig("outputs/word_freq_free_bp.png")
plt.close()
# Spam emails have a higher median and variability than ham emails, and both distributions have extreme outliers. 
# The distributions partly overlap, but spam emails most likely have higher number of word 'free' in them.

sns.boxplot(data = df, x = "spam_label", y = "char_freq_!")
plt.xlabel("Email type")
plt.ylabel("Char frequency !")
plt.savefig("outputs/char_freq_!_bp.png")
plt.close()
# Spam emails have a higher median than ham emails, though ham emails greater variability and more extreme outliers. 
# The distributions overlap, that means that there are emails from both classes which have the same amount of capital letters. 

sns.boxplot(data = df, x = "spam_label", y = "capital_run_length_total")
plt.xlabel("Email type")
plt.title("The disribution of capital letters frequency")
plt.ylabel("Total number of capital letters")
plt.savefig("outputs/capital_run_length_total_bp.png")
plt.close()
# Spam emails have a higher median than ham emails, greater variability, and more extreme outliers. 
# The distributions overlap, that means that there are emails from both classes which have the same amount of capital letters. 

# Most emails do not contain many of these words, so the data are highly skewed toward zero. 
# The features have very different numeric scales because they measure different quantities (word frequencies, character frequencies, and capital-letter counts). 
# This matters because models like logistic regression are sensitive to feature scales.

# Task 2
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
scaler = StandardScaler() # As features have different scales and we are going to use classifier which is sensitive to scale data has to be scaled first
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)
cum_expl = np.cumsum(pca.explained_variance_ratio_)
components = pca.components_ 
plt.plot(range(1, len(components) + 1), cum_expl, marker='o', linestyle='--')
plt.title("Cumulative explained variance vs. number of components")
plt.ylabel("Cumulative explained variance")
plt.xlabel("Number of components")
plt.grid(linestyle = '--', color = 'gray', alpha = 0.25, linewidth=0.5)
plt.savefig("outputs/spam_variance_explained.png")
plt.close()
n = np.argmax(cum_expl >= 0.90) + 1
print("Tne n number of components for PCA:", n)
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# Task 3
knn = KNeighborsClassifier(n_neighbors=5)
# 1st classifier
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("Accuracy knn unscaled:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 2nd classifier
knn.fit(X_train_scaled, y_train)
pred_scaled = knn.predict(X_test_scaled)
print("Accuracy knn scaled:", accuracy_score(y_test, pred_scaled))
print(classification_report(y_test, pred_scaled))

knn.fit(X_train_pca, y_train)
pred_pca = knn.predict(X_test_pca)
print("Accuracy knn pca:", accuracy_score(y_test, pred_pca))
print(classification_report(y_test, pred_pca))
# with pca used accuracy decreased a little. The model works the best on scaled data.

# 3rd classifier

for depth in [3, 5, 10, None]:
    dtc = DecisionTreeClassifier(max_depth= depth, random_state=42)
    dtc.fit(X_train, y_train)
    train_acc = dtc.score(X_train, y_train)
    test_acc = dtc.score(X_test, y_test)
    print(f"Max depth = {depth}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}\n")
# As the tree depth increases, the training accuracy increases and eventually reaches 100%. 
# The test accuracy also increases, although by a much smaller amount. 
# This suggests that, on this dataset, deeper trees improve performance and there is no strong evidence of overfitting among the depths tested.
dtc = DecisionTreeClassifier(max_depth = None, random_state = 42)
dtc.fit(X_train, y_train)
pred_final = dtc.predict(X_test)
print("Accuracy (max_depth = none):", accuracy_score(y_test, pred_final))
print(classification_report(y_test, pred_final))
dt_importance = pd.DataFrame({"feature": X_train.columns, "importance": dtc.feature_importances_})
dt_importance = dt_importance.sort_values(by="importance", ascending = False)
print("Top 10 Decision Tree features:")
print(dt_importance.head(10))

# 4th classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Accuracy Random Forest Classifier:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
rf_importance = pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_})
rf_importance = rf_importance.sort_values(by="importance", ascending = False)
top_10_features = rf_importance.head(10)
print("Top 10 Random Forest features:")
print(top_10_features)
plt.barh(top_10_features["feature"], top_10_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# 5th classifier
log_reg = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
lg_pred = log_reg.predict(X_test_scaled)
print("Accuracy logistic regression scaled:", accuracy_score(y_test, lg_pred))
print(classification_report(y_test, lg_pred))

log_reg.fit(X_train_pca, y_train)
lg_pred_pca = log_reg.predict(X_test_pca)
print("Accuracy logistic regression pca:", accuracy_score(y_test, lg_pred_pca))
print(classification_report(y_test, lg_pred_pca))
# The classifiers performed better on scaled data than on PCA-transformed data. But the difference is insignificant.
# PCA did not improve performance, possibly because the original features did not contain strong correlations or 
# because reducing the feature space removed some useful information.

# The best accuracy was achieved by the Random Forrest classifier: 0.946.

# For spam detection, accuracy alone is not the best metric to optimize. Minimizing false positives is especially important because 
# ham emails incorrectly marked as spam may cause users to miss important messages. 
# However, false negatives should also be considered because allowing spam through reduces the usefulness of the filter.
# The model achieved a spam precision of 0.92, meaning that most emails classified as spam were actually spam, reducing false positives. 
# Its spam recall was 0.90, meaning some spam messages still passed through.
cm = confusion_matrix(y_test, rf_pred)
labels = ["spam", "ham"]
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
disp.plot()
plt.title("Confusion Matrix. Spam Detection")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

# Task 4

models = [knn, dtc, rf, log_reg]

for model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Mean {model}: {cv_scores.mean():.3f}")
    print(f"Std {model}:  {cv_scores.std():.3f}")
# Random Forrest classifier is the most accurate.
# Logistic Regression Classifier is the most stable. Ranking doesn't match the single train/test split. 
# Decision Tree classifier showed lower accuracy than on single split, though Random Forrest classifier showed higher accuracy than on single split. 


# Task 5

log_reg_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))]) 
log_reg_pipeline.fit(X_train, y_train) # I am not using PCA here because it didn't improve Logistic Regression 
lgp_pipe_pred = log_reg_pipeline.predict(X_test)

rf_pipeline = Pipeline([("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]) 
rf_pipeline.fit(X_train, y_train)
rf_pipe_pred = rf_pipeline.predict(X_test)

print("Accuracy logistic regression pipeline:", accuracy_score(y_test, lgp_pipe_pred))
print("Report for logistic regression pipeline:", classification_report(y_test, lgp_pipe_pred))
print("Accuracy random forest pipeline:", accuracy_score(y_test, rf_pipe_pred))
print("Report for random forrest pipeline:", classification_report(y_test, rf_pipe_pred))

# The two pipelines do not have the same structure. The Logistic Regression pipeline includes a StandardScaler because it is sensitive to the scale of the input features.
# The Random Forest pipeline does not include scaling because tree-based models are not affected by feature scaling.

# Packaging preprocessing and the model into a pipeline ensures that the same steps are applied consistently during both training and prediction.
# This reduces the risk of errors, makes the workflow easier to reuse, and simplifies sharing or deploying the model because all preprocessing and
# prediction steps are contained in a single object.