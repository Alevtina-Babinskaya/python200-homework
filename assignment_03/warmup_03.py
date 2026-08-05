import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Preprocessing Question 1
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y, test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Question 2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.mean(axis=0))
# I fit the scale on the train data to avoid distorting data from test, in other words to avoid a data leakage.
# ------------------------------------------------------------------------------------------

# KNN Question 1
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
print("Accuracy knn:", accuracy_score(y_test, predict))
print(classification_report(y_test, predict))

# KNN Question 2
knn.fit(X_train_scaled, y_train)
predict_scaled = knn.predict(X_test_scaled)
print("Accuracy scaled:", accuracy_score(y_test, predict_scaled))
print(classification_report(y_test, predict_scaled))
# Scaling hurt performance. The accuracy fell for about 7%. Scaling reduced the natural separation that original petal measurements provide between species.

# KNN Question 3
cv_scores = cross_val_score(knn, X_train, y_train, cv = 5)
print(cv_scores)          
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
# This result is more trustworthy than a single train/test split, because we use several splits and train on different parts of data

# KNN Question 4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
# I would choose k = 5 or k = 7 because these two values give the best performance (maximum mean)
#--------------------------------------------------------------------------------------------------
# Classifier Evaluation Question 1
cm = confusion_matrix(y_test, predict)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = iris.target_names)
disp.plot()
plt.title("Confusion Matrix. Iris")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
# The model confuse no pairs of species
#--------------------------------------------------------------------------------------------------

#Decision Trees Question 1
dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(X_train, y_train)
dtc_predicted = dtc.predict(X_test)
print("Accuracy dtc:", accuracy_score(y_test, dtc_predicted))
print(classification_report(y_test, dtc_predicted))
# On unscaled data Decision Tree accuracy (97%) is lower than KNN (100%). 
# Scaling data wouldn't affect the result because Decision Tree Classifier doesn't care about data range.
#--------------------------------------------------------------------------------------------------

# Logistic Regression Question 1
c_values = [0.01, 1.0, 100]
for c in c_values:
    log_reg = LogisticRegression(C=c, max_iter = 1000)
    log_reg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(log_reg.coef_).sum()
    print(f"C value: {c}, the coefficient size: {coef_sum}")
# The total coefficient magnitude increases as C increases. 
# Regularization stabilizes the model by controlling the weight coefficients. Stronger regularization (smaller C) keeps the coefficients smaller,
# while weaker regularization (larger C) allows them to grow.

#--------------------------------------------------------------------------------------------------

# Data for PCA
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Question 1
print("X_digits: ",X_digits.shape)
print("Images: ", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for digit in range(10):
    index = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[index], cmap = 'gray_r', vmin = 0, vmax = 16)
    axes[digit].axis("off")
plt.savefig("outputs/sample_digits.png")
plt.close()

# PCA Question 2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# PCA Question 3
cum_expl = np.cumsum(pca.explained_variance_ratio_)
components = pca.components_ 
plt.plot(range(1, len(components) + 1), cum_expl, marker='o', linestyle='--')
plt.title("Cumulative explained variance vs. number of components")
plt.ylabel("Cumulative explained variance")
plt.xlabel("Number of components")
plt.grid(linestyle = '--', color = 'gray', alpha = 0.25, linewidth=0.5)
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
# To explain 80% of variance I need about 12 components

# PCA Question 4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))   

for col in range(5):  # columns corresponds to digits
    axes[0, col].imshow(images[col], cmap="gray_r", vmin=0, vmax=16)
    axes[0, col].set_title("Original")
    axes[0, col].axis("off")
    for row, n_components in enumerate([2, 5, 15, 40], start=1): # rows corresponds to numbers of components
        ax = axes[row, col]
        ax.imshow(reconstruct_digit(col, scores, pca, n_components), cmap = "gray_r", vmin = 0, vmax = 16)
        ax.set_title(f"{col} for {n_components} components")
        ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
# Some digits are clearly recognizable even with 5 components, most of them recognizable with 15 components, and using 40 components gives the best result. 
# This result matches the variance curve which shows that with 40 components we achive almost 100% explained variance, and 15 components gives about 83% of explained variance.




