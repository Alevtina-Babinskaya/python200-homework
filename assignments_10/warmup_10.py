# ML/LLM Question 1
# The ML classifier analyze and classifyes the data according rules, 
# it takes the set of rules and data and produces a boolean answer, separating data in two classes.
# The LLM produces a human-friendly recommendation based on raw data and classification. 
# ML classifier is good at analysing and classifying data, but it can not produce human-friendly language. 
# LLM is good at generating human-like language, but using it as classifier would be time-consuming and expensive.

# ML/LLM Question 2
# For each task below, write one sentence in a comment block stating whether you would use a trained ML model, an LLM, or deterministic code, and why:
# Converting a date string like "2023-07-04" to day-of-week. - deterministic code. The task is precise and there is no need for AI here.
# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text. - LLM model. 
# The task requires working with freeform text.
# Predicting customer churn given 15 numeric features and a labeled training dataset - Trained ML model, as task involves predictions based on numeric features. 
# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form. - Deterministic code. The number of variants for every city is limited,
# so the task can be done without involving AI. 
# Summing a column of revenue figures - Deterministic code, as task requires math operations and numeric data.

# ML/LLM Question 3
# Incremental processing is mechanism when the pipeline proceeds without redoing completed work.
# This mechanism prevents pipeline from redoing the work or dublicating rows.
# Completing one data record takes time and tokens. In case of large dataset, time and cost might be significant.

# Prompt Question 1
PROMPT = ("You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction about whether the day is good for running."
    "Write exactly two sentences — direct, practical, and specific to the conditions. "
    "In the first sentence state a prediction, and in the second explain the reasoning."
    "Do not use bullet points, headers, or phrases like 'Based on the data'.")
# In validation logic the code should verify that response includes exactly two sentences. 

# Prompt Question 2
import time
def call_with_retry(client, messages, max_retries=3):
    for attempt in  range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": messages},
            ]
            )
            return response
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None 
# In a production pipeline, I would use this logic when temporary API, network, or service errors might occur. 
# Retrying can allow the pipeline to recover automatically instead of failing immediately.   

   



