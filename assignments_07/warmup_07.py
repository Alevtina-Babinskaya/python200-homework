import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
import os
api_key = os.getenv("OPENAI_API_KEY")

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# --- Lesson 02 --- 
# Q1 
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time()
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"
tools_CF = [{
    'type': 'function',
    'function': {
        'name': 'celsius_to_fahrenheit',
        'description': 'Convert temperature in Celsius to the temperature in Fahrenheit',
        'parameters': {
            'type': 'object',
            'properties': {
                'celsius': {
                    'type': 'number',
                    'description': 'The temperature in degrees Celsius.'}},
            'required': ['celsius'],
        },
    },
}]
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''
# Calling run_agent("Convert 100 degrees Celsius to Fahrenheit") won't trigger a tool call because the only tool run_agent can use is current_ time function 
# which is described as returning the local current time. The model won't use it to answer the question.
# Only one API call will be made to answer this query, because no tools will be called and the model will have to answer directly.
response = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(response)
# My prediction was correct when I used tools description from the lesson. Otherwice model got confused and responded with error.

# Q3
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    },
    {
    'type': 'function',
    'function': {
        'name': 'celsius_to_fahrenheit',
        'description': 'Convert temperature in Celsius to the temperature in Fahrenheit',
        'parameters': {
            'type': 'object',
            'properties': {
                'celsius': {
                    'type': 'number',
                    'description': 'The temperature in degrees Celsius.'}},
            'required': ['celsius'],
        },
    },
}
]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time and convert Celsius degrees into Fahrenheit degrees.
                     Use the tool get_current_time whenever a user asks about the time. Use the tool celsius_to_fahrenheit whenever a user 
                     asks about converting Celsius into Fahrenheit'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                celsius = arguments['celsius']
                tool_result = celsius_to_fahrenheit(celsius)
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# Tool celsius_to_fahrenheit was called to answer the first question because the question required converting Celsius degrees into Fahrenheit degrees
# No tools were called to answer the second question, because the query was not supposed to involve converting degrees.

# ------ Lesson 03: Multi-Tool Agent -----
# Q4

# ------ CSV Manager Class --------
RESOURCES_DIR = Path("assignments_07/agent_resources")
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, file_name: str) -> str:
        if not file_name.lower().endswith(".csv"):
            return file_name + ".csv"
        return file_name
    
    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists:
            return []
        return sorted(
            [
                p.name 
                for p in self.resources_dir.iterdir()
                if p.is_file and p.suffix.lower() == ".csv"
            ]
        )
    
    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None
    
    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }
    
    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()
    
    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned
    
    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        path = f"assignments_07/outputs/{x}_vs_{y}.png" 
        plt.savefig(path)
        plt.close()
        
        return f"Plotted {y} vs {x} as a {plot_type}."
    
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
        
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {
                     "error": f"Columns must exist. Options: {self.df.columns.tolist()}"
    }
        
        pearson_r, p_value = scipy.stats.pearsonr(self.df[col1], self.df[col2])
        return {'col1': col1, 'col2': col2, 'pearson_r': round(pearson_r, 4), 'p_value': round(p_value, 4)}

print("Class defined")

# ----------- run agent cycle -------------
csv_backend = CsvManager(RESOURCES_DIR)
node_tools = {
    'list_csv_files': csv_backend.list_csv_files,
    'load_csv': csv_backend.load_csv,
    'get_columns': csv_backend.get_columns,
    'summarize_columns': csv_backend.summarize_columns,
    'describe_column': csv_backend.describe_column,
    'plot_data': csv_backend.plot_data,
    'compute_correlation': csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Calculate the Pearson's r and p-value and round both values to 4 decimal places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "Column name from dataset"},
                    "col2": {"type": "string", "description": "Column name from dataset"},
                },
                "required": ["col1", "col2"],
            }
        }
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})
    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message
    
    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
            )
        msg = response.choices[0].message
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)
        if not msg.tool_calls:
            return msg.content
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or {})
            print(f"ACT: {name}({tool_args})")
            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try: 
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
            messages.append(observe_tool_result(tool_call.id, result))
    return "I hit the tool-round limit. Try a simpler request."

#-----Q5-----
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in agent_resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

#------- Q6 ---------
# system: provides instructions and context for the agent
# user: provides the user's request or input
# assistant: represents the agent's response and tool-call decisions
# tool: represents the result returned by a tool
print(json.dumps(messages, indent=2, default=str))

# ------- Q7 ------------
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)


model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)
SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and saves the plot to assignments_07/outputs. 
        Returns a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Calculate Pearson correlation between two columns.

    Args:
        col1: The name of the first column.
        col2: The name of the second column.
    """
    return csv_manager.compute_correlation(col1=col1, col2=col2)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]


print("Function description: ", compute_correlation.description)

# ------- Q8 -------

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can work with CSV files using the provided tools and Python code.

Rules:

- Prefer the provided tools for CSV operations whenever they are sufficient.
- Use Python code when the tools are not sufficient.
- Always actually perform the requested action. Do not merely describe what you would do.
- Be honest: only claim something was done if the corresponding tool or Python code actually completed it.
- The active dataset is available as csv_manager.df after a CSV is loaded.

IMPORTANT:
When the user's request has been completed, ALWAYS finish by calling final_answer()
with a short natural-language response to the user.

Do NOT respond with ordinary natural-language prose as an intermediate step.
If there is nothing else to execute, use final_answer() to finish the task.

Plotting:
- For simple plots, prefer the plot_data tool.
- If the user requests plot styling that plot_data cannot control
  (such as color, marker, title text, labels, or grid), DO NOT use plot_data.
- Instead, write and execute matplotlib code directly using csv_manager.df.
- If Python execution fails, do not fall back to plot_data when the user specifically requested styling.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min with green dots as a scatter plot and save into assignments_07/outputs"
response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

# chat agent
chat_model = OpenAIServerModel(api_key=os.environ["OPENAI_API_KEY"], model_id="gpt-4o")
chat_agent = CodeAgent(
    model=chat_model, 
    tools=TOOLS, 
    instructions=CODE_INSTRUCTIONS, 
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=10,
    )
print("CSV analysis agent at your service. Here to help look at your CSV data!")
print("Type a question. Type 'exit' to quit.\n")
print("To start, try 'list csv files' or 'load bike_commute.csv'\n")
k = 0
while k < 5:
    user_msg = input("You: ")
    if user_msg.strip().lower() in {"quit", "exit"}:
        print("Exiting chat.")
        break
    response = chat_agent.run(user_msg, additional_args={"csv_manager": csv_manager}, reset=False)
    k += 1
print(response_code)
print(response)
# ToolCallingAgent produced plot with blue(default) dots using defined plot_data tool. 
# CodeAgent produced the plot with green dots according to the user's request.
# ToolCallingAgent can use only predefined tools, CodeAgent is able to generate code itself to tailor response to user's request.

# -------- Q9 --------
# ToolCallingAgent would be a better choice for simple, straightforward tasks that can be completed using predefined tools, 
# such as calculating an average value using a calculator tool. Tasks with clearly defined inputs, parameters, 
# and steps are a good fit for a tool-based approach because the agent only needs to select and call the appropriate 
# tool rather than generate code.
#
#One meaningful risk of using a CodeAgent is that the generated code can contain errors or unintended operations, 
# and the agent will actually execute that code. This risk does not apply in the same way to a ToolCallingAgent 
# because it calls predefined tools instead of generating and executing arbitrary cod