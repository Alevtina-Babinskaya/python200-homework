from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
import pandas as pd
from pathlib import Path
import scipy
import os
from prefect import task, flow


DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
df = None
# Tool 1
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    Returns:
        A dict with a dataset shape and column names, or an error dict.
    """
    data_path = Path(DATA_PATH)
    global df                 # the loaded data are accessible outside the function
    if data_path.exists():
        df = pd.read_csv(DATA_PATH)
    else:  
        path = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_"
        df = []
        for year in range(2015, 2025):
            data = pd.read_csv(path + str(year) + ".csv", sep = ";")
            data = data.rename(columns={"Ladder score": "Happiness score"})
            data["year"] = year
            df.append(data)
        df = pd.concat(df, ignore_index = True)  
    return {"shape": df.shape, "columns": df.columns.tolist()}

# Tool 2
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
        Args:
        column: The name of the column to describe.

        Returns:
        A dict of basic stats for the column, or an error dict.
    """
    if df is None:
        return {"error": "data are not loaded."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    else:
        return df[column].describe().to_dict()

# Tool 3
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    Args:
        col1: The name of the first column.
        col2: The name of the second column.
    Returns:
        A dictionary with two column names, pearsons r and p value
    """
    if df is None:
        return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    )
                }
        
    if col1 not in df.columns or col2 not in df.columns:
        return {
                "error": f"Columns must exist in {df.columns.tolist()}"
                }
    pearson_r, p_value = scipy.stats.pearsonr(df[col1], df[col2])
    return {'col1': col1, 'col2': col2, 'pearson_r': round(pearson_r, 4), 'p_value': round(p_value, 4)}        

# Tool 4
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

Args:
    column: The name of the column to rank countries by.
    year: The year to filter the data by.
    n: The number of top countries to return.

Returns:
    A list of dictionaries containing the country and the requested column value.
"""
    if df is None:
        return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    )
                }
    if column not in df.columns:
        return {
                "error": f"Columns must exist in {df.columns.tolist()}"
                }
    filtered_df = df[df['year'] == year]
    df_sorted = filtered_df.sort_values(column, ascending=False)
    df_top_n = df_sorted.head(n)
    return df_top_n[["Country", column]].to_dict(orient="records")




@task
def queris_task3(agent, queries):
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)
@task
def my_queries(agent):
# My query 1
    my_query_1 = "What is the correlation between social support and happiness score? Is it statistically significant?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
# Comment: This query triggered tool use only.

# My query 2
    my_query_2 = "Plot Social support vs Happiness score as a scatter plot and save into assignments_07/outputs/social_support_happiness.png"  
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
# Comment: This query triggered both tool use and code generation

# --- Reflection ---
#
# 1. In Query 3, model answered that the result was statistically significant. It calculated the p-value and rounded it to 
# 4 decimals. As it resulted in 0.0 it means it is less than 0.00005. 
#
# 2. I was surprised that it managed to plot graphics without special tools provided.
#
# 3. Plotting tool would make agent more effective. It would create plots without forsing the agent to generate code. 
# In this case the result would be more reliable. 
# The tool that loads and returns the dataset would be helpful. Agent got confused and produced errors because loading tool returns dictionary, not data.
@flow
def pipeline():
    api_key=os.environ["OPENAI_API_KEY"]
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

    SYSTEM_PROMPT = """
You are a helpful World Happiness dataset analysis assistant.

The global variable `df` stores the loaded pandas DataFrame.

IMPORTANT:
- `load_happiness_data()` loads the dataset into the global `df`.
- `load_happiness_data()` returns ONLY metadata: a dictionary containing
  `shape` and `columns`.
- The dictionary returned by `load_happiness_data()` is NOT the dataset
  and must NOT be treated as a list of rows.
- After calling `load_happiness_data()`, use the global `df` for data analysis.
- For example, use `df["year"]`, `df["Country"]`, etc.
- Do NOT write code such as:
      data = load_happiness_data()
      [row["year"] for row in data]
- If the dataset is not loaded, call `load_happiness_data()` first.
- Do not invent or guess data. Use the actual dataframe or tools.
"""

    agent = CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
        )
    queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the Happiness score column.",
    "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to assignments_07/outputs/happiness_by_region.png.",
    ]

    load_happiness_data()
    queris_task3(agent, queries)
    my_queries(agent)

if __name__ == "__main__":
    pipeline()