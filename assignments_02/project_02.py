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
st_perf_df_cleaned[["schoolsup", "internet", "higher", "activities"]] = st_perf_df_cleaned[["schoolsup", "internet", "higher", "activities"]].replace({"yes": 1, "no": 0})
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
# Parent's education level doesn't seem to play an importany role in their children success at first sight.

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
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]
X = st_perf_df_cleaned[feature_cols].values
y = st_perf_df_cleaned["G3"].values
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("test set: ", X_test_full.shape)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_predicted_full = model_full.predict(X_test_full)
print("Intercept:", model_full.intercept_)
print("RMSE:", np.sqrt(np.mean((y_predicted - y_test) ** 2)))
print("RMSE for full model :", np.sqrt(np.mean((y_predicted_full - y_test_full) ** 2)))
print("r2:", model.score(X_test, y_test))
print("r2 for full model test:", model_full.score(X_test_full, y_test_full))
print("r2 for full model train:", model_full.score(X_train_full, y_train_full))

for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# The strongest negative predictor is school support (-2.062), followed by previous failures (-1.145). The strongest positive predictors are internet
# access (+0.834), plans for higher education (+0.610), study time (+0.448), and sex (+0.453). Free time (-0.042), activities (-0.009), and parents'
# education have relatively small effects.

# The negative coefficient for school support is surprising because we might expect extra educational support to improve grades. A likely explanation is
# that school support is usually provided to students who are already struggling academically, so the variable reflects the students' initial
# performance rather than the effect of the support itself.

# The training 0.17 and test 0.15 R² values are close, suggesting the model generalizes reasonably well and is not overfitting the training data.

# If deploying this model, I would keep variables with larger coefficients, such as previous failures, school support, internet access, study time,
# plans for higher education, and sex, because they appear to have the strongest relationship with final grades.

# I would consider dropping activities, free time, travel time, and possibly parents' education, since their coefficients are close to zero and they
# appear to contribute little to the predictions.

plt.scatter(y_predicted_full, y_test_full, color = "blue")
plt.plot([0, 20], [0, 20], linestyle="--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.savefig("outputs/predicted_vs_actual_grades.png")
plt.close()
# Points above the diagonal mean the model underestimated the student's grade. Points below the diagonal mean the model overestimated the student's grade.
# The errors appear roughly uniform across grade levels, although the model has more difficulty predicting extreme values because there are fewer examples
# at the highest and lowest grades.

# The filtered dataset contains 357 students, and the test set contains 71 students.

# The model's RMSE is 2.855 points on a 0-20 grading scale. This means that, on average, the model's predictions are off by about 3 points from the
# student's actual final grade.

# The model's R² is 0.15. This means that the model explains about 15% of the variation in final grades using the selected features.

# The largest positive coefficient is internet access (+0.834), meaning that when this feature increases by one unit, the predicted final grade increases by 
# about 0.8 ( almost 1) points, holding other variables constant.

# The largest negative coefficient is school support (-2.062), meaning that when this feature increases by one unit, the predicted final grade 
# decreases by about 2 points, holding other variables constant.

# One surprising result was that school support had a negative coefficient. This may be because students receiving extra support were more likely to be
# struggling academically before receiving the intervention.

feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime", "G1"]
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
# After including G1 in features r2 significantly increased from 0.15 to 0.75. It doesn't mean G1 is causing G3 because G1 has the same relationships with other features. 
# This model is useful for predicting final grades after the first period grade is available
# To intervene earlier, educators would need to use features available before G1, such as attendance, previous failures, study habits, family support,
# engagement, and early assessments.