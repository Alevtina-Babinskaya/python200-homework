import pandas as pd
import numpy as np
from prefect import task, flow
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from scipy import stats

@task
def load_data(link, filename):
    dfs = []
    path = link + filename
    for year in range(2015, 2025):
        data = pd.read_csv(path + str(year) + ".csv", sep = ";")
        data = data.rename(columns={"Ladder score": "Happiness score"})
        data["year"] = year
        dfs.append(data)
    return pd.concat(dfs, ignore_index = True)

@task
def clean_data(df):
    df = df.map(lambda x: x.replace('"', '').replace(',', '.') if isinstance(x, str) else x)
    for col in df.columns:
        if col not in ["Country", "Regional indicator"]:
            df[col] = pd.to_numeric(df[col], errors = "coerce")
    df["Healthy life expectancy"] = df["Healthy life expectancy"].fillna(df.groupby("Country")["Healthy life expectancy"].transform("mean"))   
    return df

@task(retries=3, retry_delay_seconds=2)
def save_data(df):
    path = "assignments_01/outputs/merged_happiness.csv"
    df.to_csv(path, index = False)

@task 
def get_statistics(df):
    logger = get_run_logger()
    mean = np.mean(df["Happiness score"])
    median = np.median(df["Happiness score"])
    std = np.std(df["Happiness score"])
    mean_by_year = df.groupby("year")["Happiness score"].mean()
    mean_by_region = df.groupby("Country")["Happiness score"].mean()
    logger.info(f"mean: {mean}, median: {median}, standard deviation: {std}, mean over years: {mean_by_year}, mean over countries: {mean_by_region}")
    return {"mean": mean, "median": median, "standard deviation": std, "mean over years": mean_by_year, "mean over countries": mean_by_region}

@task
def plots(df):
    logger = get_run_logger()
    plt.hist(df["Happiness score"], color="skyblue", bins = 20)
    plt.title("Happiness score distribution")
    plt.xlabel("Happiness score")
    plt.ylabel("Frequency")
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    plt.close()
    logger.info("Histogram is created")

    df.boxplot(by="year", column="Happiness score")
    plt.title("Happiness across years")
    plt.xlabel("Year")
    plt.ylabel("Happiness score")
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    plt.close()
    logger.info("Boxplot is created")

    plt.scatter(df["Happiness score"], df["GDP per capita"], color = "green")
    plt.title("Happiness vs GPD")
    plt.xlabel("Happiness score")
    plt.ylabel("GPD per capita")
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    plt.close()
    logger.info("Scatterplot is created")

    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot = True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.title("Correlation heatmap")
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    logger.info("Heatmap is created")

@task
def hypothesis(df):
    logger = get_run_logger()
    scores_2019 = df[df["year"] == 2019]["Happiness score"]
    scores_2020 = df[df["year"] == 2020]["Happiness score"]
    t_test, p_value = stats.ttest_ind(scores_2019, scores_2020, alternative="less")
    if p_value < 0.05:
        logger.info(f"The difference in happiness score between 2019 and 2020 years is statistically significant (t-test = {t_test}, p value = {p_value})")
    else:
        logger.info(f"There is no significant difference in happiness score between 2019 and 2020 years (t-test = {t_test}, p value = {p_value})")

    scores_switzerland = df[df["Country"] == "Switzerland"]["Happiness score"]
    scores_iran = df[df["Country"] == "Iran"]["Happiness score"]
    t_test1, p_value1 = stats.ttest_ind(scores_switzerland, scores_iran, alternative="less")
    if p_value1 < 0.05:
        logger.info(f"The difference in happiness score between Switzerland and Iran is statistically significant (t-test = {t_test1}, p value = {p_value1})")
    else:
        logger.info(f"There is no significant difference in happiness score between Switzerland and Iran (t-test = {t_test1}, p value = {p_value1})")
    return {"t_test": t_test, "p_value": p_value}

@task
def correlation(df):
    logger = get_run_logger()
    df_numeric = df.drop(columns = ["Ranking", "Country", "Regional indicator", "Happiness score", "year"])
    num_tests = len(df_numeric.columns)
    corrected_alpha = 0.05 / num_tests
    corr_results = pd.DataFrame(columns=["variable", "r", "p"])
    for col in df_numeric:
        r, p = stats.pearsonr(df_numeric[col], df["Happiness score"])
        if p < 0.05:
            if p < corrected_alpha:
                logger.info(f"Pearsons r for {col} and happiness scores equals {r} with p value equals {p} which confirms that the result is statistically significant. The result is still statistically significant after alpha correction")
                corr_results.loc[len(corr_results)] = [col, r, p]
            else: 
                logger.info(f"Pearsons r for {col} and happiness scores equals {r} with p value equals {p}. The result appears statistically insignificant after checking against corrected alpha {corrected_alpha}")
        else:
            logger.info(f"Pearsons r for {col} and happiness scores equals {r} with p value equals {p} which is statistically insignificant")
    corr_results = corr_results[corr_results["r"].abs() > 0.1]
    corr_results["direction"] = corr_results["r"].apply(lambda x: "negative" if x < 0 else "positive")
    corr_results["strength"] = corr_results["r"].apply(lambda x: "weak" if abs(x) < 0.4 else "moderate" if abs(x) < 0.7 else "strong")
    for _, row in corr_results.iterrows():
        logger.info(f"There is {row['strength']} {row['direction']} correlation between Happiness and {row['variable']} (r = {row['r']:.3f}, p = {row['p']})")
    return corr_results

@task
def summary(df, stats, testing, corr_results):
    logger = get_run_logger()
    num_countries = df["Country"].nunique()
    num_years = df["year"].nunique()
    countries = stats["mean over countries"].sort_values(ascending=False)
    max_corr = corr_results.loc[corr_results["r"].idxmax()]
    logger.info(f"The dataset includes {num_countries} countries across {num_years} years.")
    logger.info(f"The contries that scored highest in happines are {countries.head(3)}, the countries that scored the lowest in happiness are {countries.tail(3)}")
    logger.info(f"There is no significant difference in happiness score between 2019 and 2020 years (t-test = {testing['t_test']}, p value = {testing['p_value']})")
    logger.info(f"{max_corr["variable"]} has the strongest correlation with happiness (r = {max_corr['r']}, p = {max_corr['p']})")

@flow
def pipeline():
    link = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/"
    filename =  "world_happiness_"
    df = load_data(link, filename)
    df = clean_data(df)
    save_data(df)
    stats = get_statistics(df)
    plots(df)
    testing = hypothesis(df)
    corr_results = correlation(df)
    summary(df, stats, testing, corr_results)

if __name__ == "__main__":
    pipeline()