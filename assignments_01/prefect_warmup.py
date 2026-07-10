import pandas as pd
import numpy as np
from prefect import task, flow

data = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task 
def create_series(arr):
    return pd.Series(arr, name = 'values')

@task 
def clean_data(series):
    return series.dropna()
@task 
def summarize_data(series):
    mean = np.mean(series)
    median = np.median(series)
    std = np.std(series)
    mode = series.mode()[0]
    return {"mean": mean, "median": median, "std": std, "mode": mode}

@flow
def data_pipeline(arr):
    series = create_series(arr)
    statistics = summarize_data(clean_data(series))
    return statistics
    

if __name__  == "__main__":
    result = data_pipeline(data)
    for key, value in result.items():
        print(f"{key}: {value}")

# For this small pipeline, Prefect is more overhead than it is worth because the workflow is simple enough to run with plain Python functions. 
# The data is small, there are only three steps, the process finishes quickly. Adding Prefect introduces extra setup, decorators (@task, @flow), and configuration
# that takes more effort than writing and running the pipeline directly.

# Prefect is useful for larger or repeated workflows. For example, it can help process large amounts of data, run pipelines on a schedule, retry failed tasks, 
# track what happened during a run, manage complex workflows, and make it easier for teams to work together.