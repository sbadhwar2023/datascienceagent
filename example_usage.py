"""Example usage of the modular Data Science Agent.

Run this script from the command line to see how the agent plans and
executes an analysis. By default it uses OpenRouter with the
``mistral-large-2402`` model; set the ``--provider`` and ``--model``
flags to override.

Example:

```bash
python example_usage.py --provider litellm --model gpt-3.5-turbo
```
"""

from __future__ import annotations

import argparse
import pandas as pd
import os
from data_science_agent import DataScienceAgent


def build_sample_data() -> pd.DataFrame:
    """Return a small DataFrame for demonstration purposes."""
    return pd.DataFrame({
        'Ion Species': ['Boron', 'Phosphorus', 'Boron', 'Arsenic'],
        'Implant Energy (keV)': [20, 40, 15, 50],
        'Dose (ions/cm²)': [1e15, 5e14, 1e14, 1e15],
        'Beam Current (mA)': [1.0, 1.5, 0.8, 2.0],
        'Wafer Temp (°C)': [25, 30, 20, 35],
        'Tilt Angle (°)': [7, 5, 4, 7],
        'Uniformity (%)': [95, 93, 97, 92],
        'Depth (nm)': [120, 300, 100, 400],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Data Science Agent example.")
    parser.add_argument("--provider", choices=["openrouter", "litellm", "anthropic"], default="openrouter", help="LLM provider")
    parser.add_argument("--model", default="mistral-large-2402", help="Model name for the provider")
    parser.add_argument("--api-key", default=None, help="API key for the provider; otherwise read from environment")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv({
        "openrouter": "OPENROUTER_API_KEY",
        "litellm": "LITELLM_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[args.provider])

    if api_key is None:
        raise SystemExit("Please provide an API key via --api-key or environment variable.")

    # Create the agent
    agent = DataScienceAgent(
        llm_provider=args.provider,
        llm_api_key=api_key,
        model_name=args.model,
        llm_kwargs={"temperature": 0.1, "max_tokens": 3000},
    )

    # Sample objective and data
    objective = (
        "I have ion implantation experiment data. Analyse the relationships between parameters and implantation depth. "
        "Suggest the next set of parameters to optimise performance, aiming for minimal depth while maintaining high uniformity (>94%)."
    )
    data = build_sample_data()

    # Create and execute plan
    plan = agent.create_plan(objective, data)
    agent.execute_plan_sync()
    report = agent.get_execution_report()
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    # Optionally save results
    agent.save_results("results.json")
    print("Results saved to results.json")


if __name__ == "__main__":
    main()