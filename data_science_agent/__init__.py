"""Data Science Agent package.

This module exposes the primary `DataScienceAgent` class along with
supporting data structures for building and executing data‐driven
plans. The package is organised so that each piece of the agent’s
functionality lives in its own module. See the README in the root
folder for an overview of how to use these classes.

The main entry point for consumers is :class:`DataScienceAgent`,
which orchestrates plan generation, code execution and analysis.

Usage example:

```python
from data_science_agent import DataScienceAgent
import pandas as pd

# Create some example data
df = pd.read_csv("my_data.csv")

agent = DataScienceAgent(llm_provider="openrouter", llm_api_key="sk-…")
plan = agent.create_plan("Analyse my dataset", df)
agent.execute_plan_sync()
print(agent.get_execution_report())
```

See :mod:`data_science_agent.plan` for data models and
:mod:`data_science_agent.llm_interface` for more information on
supported LLM providers.
"""

from .agent import DataScienceAgent
from .plan import TaskStatus, SubTask, Task, Plan

__all__ = [
    "DataScienceAgent",
    "TaskStatus",
    "SubTask",
    "Task",
    "Plan",
]