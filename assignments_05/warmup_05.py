from dotenv import load_dotenv
from openai import OpenAI
import json

# The Chat Completions API
# API Question 1
load_dotenv()
client = OpenAI()
model = "gpt-4o-mini"

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}],
    n=1,
    temperature=0.7
)
print("Response from AI: ", response.choices[0].message.content)
print("Model responded: ", response.model)
print("Token used: ", response.usage.total_tokens)

# API Question 2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"Response for temperature equals to {temp}: {response.choices[0].message.content}")
# Temperature 0 produced the most consistent and obvious name. Temperature 0.7 generated a bit more varied suggestions.
# Temperature 1.5 produced the most detailed and creative name which was less consistent. If I needed reproducible output, I would use 0 temperature.

# API Question 3
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
i=1
for resp in response.choices:
    print(f"The response number {i}: {resp.message.content}")
    i += 1

# API Question 4
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens = 15
)
print(response.choices[0].message.content)
# The response is limited to 15 words, and it's too short to answer the question properly. In real applications I might want to use max_tokens to keep the coat under control.


# System Messages and Personas
# System Question 1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens= 300
)
print("The response from Python tutor", response.choices[0].message.content)
messages1 = [
    {"role": "system", "content": "You are a middle school teacher explaining Python to a student. Use simple analogies, avoid technical words, and explain like the student is 12 years old."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}

]
response = client.chat.completions.create(
    model=model,
    messages=messages1,
    max_tokens= 300
)
print("The response from middle school teacher", response.choices[0].message.content)
# The style and the level of complexity has changed dramatically. In the first response there are plenty of coding terminology, second responce is plain and creative.

# System Question 2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=300
)
print("assistant: ", response.choices[0].message.content)
# The model knows Jordan's name, because this information is included in the messages that I passed to the model.

# Prompt Engineering
# Prompt Question 1 — Zero-Shot
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
print("Zero-shot prompt")
for i, r in enumerate(reviews, 1):
    prompt = f"Classify {r} as positive, negative, or mixed."
    print(f"Review #{i}: {get_completion(prompt)}")
    i+= 1

# Prompt Question 2 — One-Shot
print("One-shot prompt")
prompt = f"Classify {reviews} as positive, negative, or mixed. Example: Review: Fast shipping but the item arrived damaged. Sentiment: mixed"
print(get_completion(prompt))
# Adding one example changed the format of the output compared to Q1.

# Prompt Question 3 — Few-Shot
print("Few-shot prompt")
prompt = f"""Classify {reviews} as positive, negative, or mixed. 
Example 1: Review: Fast shipping but the item arrived damaged. Sentiment: mixed. 
Example 2: Review: The service was extremely good, the staff was helpful. Sentiment: Positive. 
Example 3: Review: The software came without essential access codes, I had to spend hours on phone trying to fix it. Sentiment: Negative"""
print(get_completion(prompt))
# In all cases model classified reviews correctly. One example used in the second case changed the format of the response. Addin another two examples haven't changed anything.
# I would choose one-shot approach if the question is direct, the answer is obvious and the format is unimportant. 
# If the task is complicated and require an additional guidance or if the format is important, using one or few examples would be helpful.

# Prompt Question 4 — Chain of Thought
prompt = """Solve the problem showing your reasoning step by step and clearly label the final answer: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?"""
print(get_completion(prompt))
# Asking the model to reason step by step tend to improve accuracy on problems like this because steps clearly show the logic behind the answer. A user can check every step 
# for mistakes and ask questions if something needs further clarification.

# Prompt Question 5 — Structured Output
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f"""Analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence).
Review: {review}"""
resp_data = get_completion(prompt)
print(resp_data)
try:
    data = json.loads(resp_data)
    print("Sentiment: ", data['sentiment'])
    print("Confidence: ", data['confidence'])
    print("Reason: ", data['reason'])
except json.JSONDecodeError:
    print("The response is not valid JSON.")
    print(resp_data)

# Prompt Question 6 — Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
print(get_completion(prompt))

user_text_ = "My uncle’s goodness is extreme, If seriously he hath disease; He hath acquired the world’s esteem And nothing more important sees; A paragon of virtue he!"
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_}```
"""
print(get_completion(prompt))
# Delimiters help devide a long prompt into meaningful parts so model would not get confused with complex instructions.

# Local Models with Ollama
# Ollama Question 1
prompt = "Explain what a large language model is in two sentences."
print("Open AI response: ", get_completion(prompt))

"""Ollama response: A large language model is a type of artificial intelligence that can understand and generate text, trained on vast amounts of human language. 
It excels at processing and generating natural language, making it invaluable for tasks like writing, communication, and content creation."""


# The responce from Ollama is simplier and focuses on usage of LLM. The responce from Open AI contains more terms and describes the way LLM works. Also, Ollama showed 
# reasoning behind the response.

# The advantage of running a model locally is that there are no API fees.
# The disadvantages are that it uses your computer's resources and the model must be updated manually from time to time.