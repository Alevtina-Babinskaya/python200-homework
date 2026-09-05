import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Task 1
# I need to specify the separator for columns and data
st_perf_df = pd.read_csv("student_performance_math.csv", sep=";") 
print(st_perf_df.shape)
print(st_perf_df.head())
print(st_perf_df.info())
plt.hist(st_perf_df["G3"], bins=21, color="skyblue")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Math Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("outputs/g3_distribution.png")
plt.close()

# Task 2: Preprocess the Data
print(st_perf_df.shape)
st_perf_df_cleaned = st_perf_df[st_perf_df["G3"] != 0] # Keeping these rows would distort the model, because they don't actually present the real grades
print(st_perf_df_cleaned.shape)
cols = ["schoolsup", "internet", "higher", "activities"]
st_perf_df_cleaned.loc[:, cols] = (st_perf_df_cleaned.loc[:, cols].replace({"yes": 1, "no": 0}))
st_perf_df_cleaned["sex"] = st_perf_df_cleaned["sex"].replace({"F": 0, "M": 1})
print(st_perf_df_cleaned.head())
r1 = st_perf_df["G3"].corr(st_perf_df["absences"])
r2 = st_perf_df_cleaned["G3"].corr(st_perf_df_cleaned["absences"])
print(f"Correlation between final grade and number of absences before cleaning was {r1}, and after cleaning is {r2}")
# The students with 0 grade might have different number of absence. There is no correlation here. As the number of absence is => 0 for students who has 0 grade, 
# it increases overall correlation coefficient to make it positive which doesn't make sense.  

# Task 3: Exploratory Data Analysis
df_numeric = st_perf_df_cleaned[["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime", "goout", "Walc"]]
corr_results = pd.DataFrame(columns = ["feature", "r"])
for col in df_numeric:
    r = st_perf_df_cleaned["G3"].corr(df_numeric[col])
    corr_results.loc[len(corr_results)] = [col, r]
corr_results = corr_results.sort_values(by="r", ascending = True)
for _, row in corr_results.iterrows():
    print(f"The correlation between {row['feature']} and final grade is {row['r']}.")
# The strongest negative correlation is observed between number of failures and final grade, which seems natural. 
# The strongest positive correlation is obseved between mother's education level and final grade. This relationship is surprising. 
# Parent's education level doesn't seem to play an important role in their children success at first sight.

# Students with more previous failures generally earn lower final grades.
plt.scatter(st_perf_df_cleaned["failures"], st_perf_df_cleaned["G3"], color = "green", alpha=0.5)
plt.title("Number of Failures vs Students Final Period Grade")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade")
plt.savefig("outputs/failures_vs_G3.png")
plt.close()

# Most students have relatively few absences, and there is no strong linear
# relationship between absences and final grades. Students with both high and
# low grades appear across the range of absence counts.
plt.scatter(st_perf_df_cleaned["absences"], st_perf_df_cleaned["G3"], color = "green", alpha=0.5)
plt.title("Number of absences vs Students Final Period Grade")
plt.xlabel("Number of absences")
plt.ylabel("Final Grade")
plt.savefig("outputs/absences_vs_G3.png")
plt.close()

# Students who study more hours per week tend to have slightly higher final
# grades, although the grade distributions overlap across study-time groups.
plt.figure(figsize=(6, 4))
st_perf_df_cleaned.boxplot(column="G3", by="studytime")
plt.title("Final Grade by Weekly Study Time")
plt.suptitle("")   
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/studytime_boxplot.png")
plt.close()

# Task 4: Baseline Model
X = np.array(st_perf_df_cleaned["failures"]).reshape(-1, 1)
y = st_perf_df_cleaned["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", np.sqrt(np.mean((y_predicted - y_test) ** 2)))
print("r2:", model.score(X_test, y_test))
# Slope means how much grade changes for every failures category. The slop is negative so as the number of failures increases, the grade decreases.
# The RMSE tells us the model's typical prediction error. The error for this model is close to 3 points.
# r2 shows how good the model predicts the result. This r2 is low, that means that model doesn't fit well the data.

# Task 5: Build the Full Model
feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = st_perf_df_cleaned[feature_cols].values
y = st_perf_df_cleaned["G3"].values
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("test set: ", X_test_full.shape)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_predicted_full = model_full.predict(X_test_full)
print("Full model")
print("Intercept:", model_full.intercept_)
print("RMSE:", np.sqrt(np.mean((y_predicted - y_test) ** 2)))
print("RMSE for full model :", np.sqrt(np.mean((y_predicted_full - y_test_full) ** 2)))
print("r2:", model.score(X_test, y_test))
print("r2 for full model test:", model_full.score(X_test_full, y_test_full))
print("r2 for full model train:", model_full.score(X_train_full, y_train_full))

for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# The strongest negative predictor is school support (-2.263), followed by previous failures (-0.800) and going out (-0.313). 
# The strongest positive predictors are internet access (+1.037), sex (+0.402), study time (+0.311), and parents' education, 
# particularly father's education (+0.187) and mother's education (+0.163). Free time (+0.014), activities (+0.061), and 
# travel time (-0.083) have relatively small effects.

# The negative coefficient for school support is surprising because we might expect extra educational support to improve grades. 
# A likely explanation is that school support is usually provided to students who are already struggling academically, so 
# the variable may reflect students' initial academic difficulties rather than the effect of the support itself. 
# The negative coefficient for previous failures (-0.800) is also consistent with the expectation that students with 
# more previous failures tend to have lower final grades.

# The training R² (0.235) and test R² (0.263) are relatively close, suggesting that the model generalizes reasonably well and 
# is not overfitting the training data. However, the relatively low R² values indicate that the model explains only a limited 
# portion of the variation in final grades.

# If deploying this model, I would pay particular attention to variables with larger coefficients, such as school support, 
# previous failures, internet access, going out, sex, and study time, because they show the strongest relationships with final grades.

# I would consider dropping free time, activities, travel time, and possibly parents' education, since their coefficients are 
# relatively small and they appear to contribute less to the predictions. However, coefficient size alone should not be the only 
# criterion for removing variables.

plt.scatter(y_predicted_full, y_test_full, color = "blue")
plt.plot([0, 20], [0, 20], linestyle="--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.savefig("outputs/predicted_vs_actual_g3.png")
plt.close()

# Points above the diagonal mean that the model underestimated the student's grade. Points below the diagonal mean that the model overestimated the student's grade.
# The errors appear roughly distributed across grade levels, although the model may have more difficulty predicting extreme values because there are fewer examples
# at the highest and lowest grades.
# The filtered dataset contains 357 students, and the test set contains 72 students.
# The model's RMSE is 2.96 points on a 0-20 grading scale. This means that the model's predictions are typically off by about 3 points from the
# student's actual final grade.
# The model's R² is 0.089. This means that the model explains about 9% of the variation in final grades using the selected features.
# The largest positive coefficient is internet access (+1.037), meaning that a one-unit increase in this feature is associated with an increase of about
# 1.04 points in the predicted final grade, holding other variables constant.
# The largest negative coefficient is school support (-2.263), meaning that a one-unit increase in this feature is associated with a decrease of about
# 2.26 points in the predicted final grade, holding other variables constant.
# One surprising result was that school support had a negative coefficient. This may be because students receiving extra support were more likely to be
# struggling academically before receiving the intervention. Therefore, the coefficient may reflect pre-existing academic difficulties rather than the
# effect of school support itself.


feature_cols_g1 = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex", "G1"]
X = st_perf_df_cleaned[feature_cols_g1].values
y = st_perf_df_cleaned["G3"].values
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("test set: ", X_test_full.shape)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_predicted_full = model_full.predict(X_test_full)
print("Intercept g1:", model_full.intercept_)
print("RMSE for full model with g1 :", np.sqrt(np.mean((y_predicted_full - y_test_full) ** 2)))
print("r2 for full model test g1:", model_full.score(X_test_full, y_test_full))
# After including G1 in features r2 significantly increased from 0.23 to 0.76. It doesn't mean G1 is causing G3 because G1 has the same relationships with other features. 
# This model is useful for predicting final grades after the first period grade is available
# To intervene earlier, educators would need to use features available before G1, such as attendance, previous failures, study habits, family support,
# engagement, and early assessments.