import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Pandas Q1
print(f"First 3 rows: {df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Data types: {df.dtypes}")

# Pandas Q2
print(df[(df["passed"] == True) & (df["grade"] > 80)])

# Pandas Q3
df["grade_curve"] = df["grade"] + 5
print(df)

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5
print(df.groupby("city")["grade"].mean())

# Pandas Q6
df["city"] = df["city"].replace({"Austin": "Houston"})
print(df[["name", "city"]])

# Pandas Q7
df = df.sort_values("grade", ascending=False)
print(df.head(3))

# NumPy Q1
arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}")
print(f"Data types: {arr.dtype}")
print(f"Dimensions: {arr.ndim}")

# NumPy Q2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3
print(arr[0:2, 0:2])

# NumPy Q4
arr_zeros = np.zeros((3, 4))
print(arr_zeros)
arr_ones = np.ones((2, 5))
print(arr_ones)

# NumPy Q5
arr1 = np.arange(0, 50, 5)
print(arr1)
print(f"Mean: {np.mean(arr1)}")
print(f"Shape: {np.shape(arr1)}")
print(f"Sum: {np.sum(arr1)}")
print(f"Standard deviation: {np.std(arr1)}")

# NumPy Q6
arr_normal = np.random.normal(0, 1, 200)
print(np.mean(arr_normal))
print(np.std(arr_normal))

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x,y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Subject Scores" )
plt.xlabel("subjects")
plt.ylabel("scores")
plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color = "blue", label="Dataset 1")
plt.scatter(x2, y2, color = "orange", label="Dataset 2")
plt.title("Dataset comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q4
fig, ax = plt.subplots(1, 2)
ax[0].plot(x, y)
ax[0].set_title("Squares")
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")
plt.tight_layout()
plt.show()

# Descriptive Stats Question 1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard deviation: {np.std(data)}")

# Descriptive Stats Question 2
normal_data = np.random.normal(65, 10, 500)
plt.hist(normal_data, bins=30, color="skyblue", edgecolor="black")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Question 4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
fig, axes = plt.subplots(1, 2)
axes[0].hist(normal_data, bins=30, color="skyblue", edgecolor="black")
axes[0].set_title("Normal Distribution")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(skewed_data, bins=30, color="pink", edgecolor="black")
axes[1].set_title("Exponential Distribution")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")
fig.suptitle("Distribution Comparison")
plt.tight_layout()
plt.show()

# Descriptive Stats Question 5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print(f"Data 1: Mean - {np.mean(data1)}, Median - {np.median(data1)}, Mode - {stats.mode(data1)}")
print(f"Data 2: Mean - {np.mean(data2)}, Median - {np.median(data2)}, Mode - {stats.mode(data2)}")
# In data2 median is so different from mean because data contains an outlier 150 that skews average value

# Hypothesis Question 1 
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_test, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-test: {t_test}")
print(f"P value: {p_value}")

# Hypothesis Question 2 
if p_value < 0.05:
    print("The result is statistically significant")
else:
    print("The result is statistically unsignificant")

# Hypothesis Question 3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_test, p_value = stats.ttest_rel(before, after)
print(f"T-test: {t_test}")
print(f"P value: {p_value}")

# Hypothesis Question 4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_test, p_value = stats.ttest_1samp(scores, 70)
print(f"T-test: {t_test}")
print(f"P value: {p_value}")

# Hypothesis Question 5
t_test, p_value = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"P value one tailed: {p_value}")

# Hypothesis Question 6
print("The results suggest that Group B has a higher average score than Group A, and this difference is very unlikely to be due to chance.")

# Correlation Question 1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print(f"Correlation: {corr_matrix}")
print(corr_matrix[0, 1]) # Looking at the data I expected strong positive correlation

# Correlation Question 2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
r, p = pearsonr(x, y)
print(f"Pearson's r = {r}, p-value = {p}")

# Correlation Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr_matrix_people = df.corr()
print(corr_matrix_people)

# Correlation Question 4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, color ="skyblue")
plt.title("Negative Correlation")
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()

# Correlation Question 5
sns.heatmap(corr_matrix_people, annot = True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pipeline Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
def create_series(arr):
    return pd.Series(arr, name = 'values')

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    mean = np.mean(series)
    median = np.median(series)
    std = np.std(series)
    mode = series.mode()[0]
    return {"mean": mean, "median": median, "std": std, "mode": mode}

def data_pipeline(arr):
    series = create_series(arr)
    return summarize_data(clean_data(series))

statistics =  data_pipeline(arr)
for key, value in statistics.items():
    print(f"{key}: {value}")


