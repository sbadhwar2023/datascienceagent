# Using the Data Science Agent

This document describes how to use the refactored data science agent
within different environments such as Jupyter notebooks and regular
Python scripts. It also covers how to select an LLM provider and
model, how to create a plan and how to run it.

## Choosing a provider and model

Three provider options are supported out of the box:

1. **openrouter** – Access models hosted on
   [OpenRouter.ai](https://openrouter.ai). Requires the `openrouter`
   Python client. Set `OPENROUTER_API_KEY` in your environment.
2. **litellm** – Use the unified API provided by
   [LiteLLM](https://github.com/BerriAI/litellm). Supports models from
   OpenAI, Anthropic, Cohere, etc. Set the appropriate API key
   (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
3. **anthropic** – Directly utilise Anthropic via the `langchain`
   wrapper. Requires `langchain-anthropic` and `ANTHROPIC_API_KEY`.

Select the provider via the `llm_provider` argument when creating
`DataScienceAgent` and pass the model name via `model_name`. For
example:

```python
agent = DataScienceAgent(
    llm_provider="openrouter",
    llm_api_key="sk-your-openrouter-key",
    model_name="mistral-large-2402",
    llm_kwargs={"temperature": 0.2, "max_tokens": 3000},
)
```

## Running in a Jupyter notebook

In notebooks you can use the synchronous API so that calls block until
completion. This avoids issues with nested event loops that occur when
calling `asyncio.run()` inside Jupyter. Example:

```python
from data_science_agent import DataScienceAgent
import pandas as pd

df = pd.read_csv("my_dataset.csv")
agent = DataScienceAgent(llm_provider="litellm", llm_api_key="sk-…", model_name="gpt-3.5-turbo")
plan = agent.create_plan("Analyse trends in sales", df)
agent.execute_plan_sync()
report = agent.get_execution_report()
print(report)
```

## Running as a script

When run from a plain Python script you can choose between the
synchronous and asynchronous APIs. To use the async version, wrap
your calls inside an `async def` function and run it with
`asyncio.run()`:

```python
import asyncio
from data_science_agent import DataScienceAgent
import pandas as pd

async def run_agent():
    df = pd.read_csv("data.csv")
    agent = DataScienceAgent(llm_provider="anthropic", llm_api_key="sk-…", model_name="claude-3-sonnet-20240229")
    plan = agent.create_plan("Optimise manufacturing yield", df)
    await agent.execute_plan()
    print(agent.get_execution_report())

if __name__ == "__main__":
    asyncio.run(run_agent())
```

## Inspecting intermediate results

During execution the agent stores variables defined by your code in
the sandbox environment. You can inspect them via
`agent.sandbox.get_context()` after running subtasks. This is useful
for debugging or for manually plotting data:

```python
context = agent.sandbox.get_context()
for name, value in context.items():
    print(name, type(value))
```

## Saving results

Call `agent.save_results("results.json")` after execution to persist
the final context, the execution report and the final
recommendations. You can then load and analyse this JSON in another
program.