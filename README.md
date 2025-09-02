# Data Science Agent

This repository contains a modular data science agent that leverages
large language models (LLMs) to plan, execute and analyse data
experiments. The agent automatically decomposes a high‑level objective
into a series of tasks and subtasks, generates Python code to perform
each step, executes it in a secure sandbox and synthesises findings
into actionable recommendations.

The code has been refactored to emphasise modularity and clarity. Each
major concern is encapsulated in its own module:

- `data_science_agent/llm_interface.py` – lightweight adapters for
  different LLM providers such as OpenRouter, LiteLLM and Anthropic.
- `data_science_agent/plan.py` – data models (dataclasses and
  Pydantic schemas) for plans, tasks and subtasks.
- `data_science_agent/sandbox.py` – a secure environment for running
  dynamically generated code with support for pandas, NumPy,
  Matplotlib and Seaborn.
- `data_science_agent/plan_generator.py` – functions for asking the
  LLM to produce a structured execution plan from an objective and
  context.
- `data_science_agent/code_generator.py` – functions for requesting
  executable code to accomplish a subtask.
- `data_science_agent/reasoning_engine.py` – functions for analysing
  intermediate results and producing final recommendations.
- `data_science_agent/agent.py` – high‑level orchestrator that ties
  everything together and exposes a simple API for users.

## Installation

Create a Python virtual environment and install the package and its
dependencies. At minimum you need to install `langchain`,
`pydantic`, `pandas` and either `openrouter` or `litellm` depending on
the provider you wish to use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn pydantic langchain

# Choose one of the following:
pip install openrouter      # for OpenRouter support
# or
pip install litellm         # for LiteLLM support

# Optional: install langchain-anthropic if you want to fall back to
# Anthropic directly (for the `anthropic` provider option).
pip install langchain-anthropic
```

## Configuration

The agent uses API keys to communicate with the chosen LLM provider.
You can either pass the key explicitly when creating the agent or set
the appropriate environment variable.

| Provider    | API key environment variable         |
|-------------|-------------------------------------|
| openrouter  | `OPENROUTER_API_KEY`                |
| litellm     | `LITELLM_API_KEY`, `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` depending on model |
| anthropic   | `ANTHROPIC_API_KEY`                 |

For example, to use OpenRouter with the model `mistral-large-2402`:

```bash
export OPENROUTER_API_KEY="sk-..."
```

Then instantiate the agent with `llm_provider="openrouter"` and the
desired `model_name`.

## Quick start

An example script is provided in `example_usage.py`. It demonstrates
creating a plan from a simple dataset, executing it and printing a
report. Run it like so:

```bash
python example_usage.py
```

You should see the agent generate a plan, run through several
analysis steps and finish with a set of final recommendations.

## Overview of classes

- **`DataScienceAgent`** – The main entry point for users. Handles
  planning, execution, and reporting.
- **`PlanGenerator`** – Uses an LLM to break down an objective into
  tasks and subtasks.
- **`CodeGenerator`** – Requests Python code from the LLM to fulfil a
  subtask.
- **`ReasoningEngine`** – Interprets results and synthesises final
  recommendations.
- **`CodeSandbox`** – Executes generated Python code in an isolated
  environment, capturing output and errors.

## Customising the agent

The prompts used to generate plans, code and analysis are embedded
within the generator classes. You can subclass `PlanGenerator`,
`CodeGenerator` or `ReasoningEngine` to override the prompts or
incorporate domain‑specific guidance. Alternatively you can adjust the
model parameters (temperature, max tokens, etc.) by passing
``llm_kwargs`` to the `DataScienceAgent`.

## License

MIT License

## Acknowledgement

This code has been created with the help of Claude Code and GPT-5