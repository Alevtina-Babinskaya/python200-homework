import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

#scikit-learn Question 1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4, 8]).reshape(-1, 1)
model = LinearRegression()
model.fit(years, salary)
predicted = model.predict(years)
predicted_salaries = model.predict(new_years)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("The salary for someone with 4 years of experience:", predicted_salaries[0])
print("The salary for someone with 8 years of experience:", predicted_salaries[1])

#scikit-learn Question 2
x = np.array([10, 20, 30, 40, 50])
X = x.reshape(-1, 1)
print("x:", x.shape)
print("X:", X.shape)
# Scikit-learn needs 2D array, so it can understand what data is supposed to be rows, means samples, and what data is supposed to be columns, means variables. 
# In case it 1D array, the list of numbers can present either samples or features.

#scikit-learn Question 3
X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X_clusters)

labels = kmeans.predict(X_clusters)

print("Cluster centers", kmeans.cluster_centers_)
print("Number of points", np.bincount(labels))

plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap="viridis",
    s=60,
    alpha=0.7
)

# Plot cluster centers as black X's
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    c="black",
    s=200
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# Linear regression
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Question 1
plt.scatter(age, cost, c = smoker, cmap = "coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("outputs/cost_vs_age.png")
# There are 2 distinct groups visible on the plot. Smokers seem to have higher medical costs than non-smokers for the same values of age.
# So smoker variable contributes to the cost.

# Linear Regression Question 2
X = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, cost, test_size = 0.2, random_state = 42)
print("X_train array:", X_train.shape)
print("X_test array:", X_test.shape)
print("y_train array:", y_train.shape)
print("y_test array:", y_test.shape)

# Linear Regression Question 3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", np.sqrt(np.mean((y_pred - y_test) ** 2)))
print("r2:", model.score(X_test, y_test))

# The slope means that the cost increases by the slope value for each additional year of age.
# r2 is very low because data falls into two groups smokers and non-smokers, and we don't use this variable here.

# Linear Regression Question 4
X_full = np.column_stack([age, smoker])
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, cost, test_size = 0.2, random_state = 42)
model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
y_full_pred = model_full.predict(X_full_test)
print("Age cofficient:", model_full.coef_[0])
print("Smoke cofficient:", model_full.coef_[1])
print("r2 full:", model_full.score(X_full_test, y_full_test))
# Adding smoke variable increases r2 significantly.
# The smoker coefficient represents how much the predicted medical cost increases for smokers compared to non-smokers of the same age.

# Linear Regression Question 5
plt.scatter(y_full_pred, y_full_test, color = "blue", label = "Data")
plt.plot([y_full_test.min(), y_full_test.max()],[y_full_test.min(), y_full_test.max()], color="red")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
# A point is above the line means that actual cost is greater than predicted
# A point is below the line means that predicted cost is greater than actual