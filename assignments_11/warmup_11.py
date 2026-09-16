from prefect import flow, task
from prefect.logging import get_run_logger

# Prefect Orchestration
# ----------------------------------
# Prefect Question 1
# @task is a decorator for a particular part of the pipeline, for example, loading the raw data from the sourse. 
# @flow is a decorator for the pipeline structure, it defines the workflow and runs the tasks in the required order. 
# You have a helper function that converts a temperature from Celsius to Fahrenheit — a pure, in-memory calculation with no I/O. 
# Would you decorate it with @task? Why or why not?
# It depends on whether I want it to be a part of pipeline flow or not. Given that this is a simple helper function I would rather not decorate it as a task.
# 
# Prefect Question 2
@task(retries = 3, retry_delay_seconds = 30)
def call_api() -> list:
    return None
# Prefect Question 3
# I will look in the logs of transform task to find the exception traceback.

# Production Patterns
# Production Question 1
# raise_for_status() raises an exception when the request returns an HTTP error status, such as 404 or 500. 
# It is better than simply writing if response.status_code != 200: print("error") in a pipeline task 
# because raising an exception causes the Prefect task to fail and provides an error traceback in the logs.
# If we only print an error message, the problem is detected but the task may continue running, 
# and the pipeline may proceed with an invalid or empty response.

# Production Question 2
# `upsert` protects us from duplicating data and allows us to update existing rows instead of failing when the same record already exists.
# INSERT` only adds a new row; it does not update an existing row. If we use `INSERT` instead of 'upsert', 
# we may either create duplicate data or get an error if the ID or another column has a unique constraint.

# Production Question 3
@task
def log_number(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(f"{len(enrichment_records)} records were upserted")
    if not enrichment_records:
        logger.warning("No enrichment records were loaded")

# Production Question 4
# Incremental processing checks how many records from the raw data have already been enriched. 
# If we remove it, the pipeline will process all raw records every time it runs, which will waste time and money.
