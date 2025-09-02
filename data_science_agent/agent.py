"""High level Data Science Agent orchestration.

The :class:`DataScienceAgent` combines plan generation, code
execution, result analysis and recommendation synthesis into a single
interface. It is designed to be flexible: you can specify which
language model provider to use (OpenRouter, LiteLLM or Anthropic) and
which model name to target. The agent persists intermediate variables
in a sandbox so that subsequent subtasks can access previous results.

Example usage::

    from data_science_agent import DataScienceAgent
    import pandas as pd

    df = pd.read_csv("data.csv")
    agent = DataScienceAgent(llm_provider="litellm", llm_api_key="sk-…", model_name="gpt-4")
    plan = agent.create_plan("Analyse sales data", df)
    agent.execute_plan_sync()
    print(agent.get_execution_report())
    agent.save_results("results.json")
"""

from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from .llm_interface import get_llm, BaseLLM
from .plan import Plan, Task, SubTask, TaskStatus
from .sandbox import CodeSandbox
from .plan_generator import PlanGenerator
from .code_generator import CodeGenerator
from .reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)


class DataScienceAgent:
    """Main class coordinating plan execution and analysis."""

    def __init__(
        self,
        llm_provider: str = "anthropic",
        llm_api_key: str | None = None,
        model_name: str = "claude-3-sonnet-20240229",
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the DataScienceAgent.

        :param llm_provider: one of ``openrouter``, ``litellm`` or ``anthropic``
        :param llm_api_key: API key for the chosen provider; if ``None`` the
            key will be read from environment variables as defined by the
            underlying adapter
        :param model_name: model identifier for the selected provider
        :param llm_kwargs: optional extra parameters passed through to the
            LLM client (e.g. temperature, max_tokens)
        """
        if llm_api_key is None:
            # Attempt to read from environment variables based on provider
            import os
            if llm_provider == "openrouter":
                llm_api_key = os.environ.get("OPENROUTER_API_KEY")
            elif llm_provider == "litellm":
                # fallback for OpenAI or Anthropic keys handled in adapter
                llm_api_key = os.environ.get("LITELLM_API_KEY")
            else:  # anthropic
                llm_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if llm_api_key is None:
                raise ValueError(
                    "llm_api_key must be provided or set in the environment for the selected provider"
                )
        self.llm: BaseLLM = get_llm(llm_provider, model_name, llm_api_key, **(llm_kwargs or {}))
        self.sandbox = CodeSandbox()
        self.plan_generator = PlanGenerator(self.llm)
        self.code_generator = CodeGenerator(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        # Execution state
        self.plans: List[Plan] = []
        self.current_plan: Plan | None = None
        self.execution_history: List[Dict[str, Any]] = []

    def _prepare_context(self, objective: str, initial_data: Any | None) -> Dict[str, Any]:
        """Prepare the context dictionary used when generating a plan."""
        context: Dict[str, Any] = {"objective": objective}
        if initial_data is not None:
            if isinstance(initial_data, pd.DataFrame):
                context.update({
                    "data_shape": initial_data.shape,
                    "columns": list(initial_data.columns),
                    "dtypes": {k: str(v) for k, v in initial_data.dtypes.to_dict().items()},
                    "sample": initial_data.head().to_dict(),
                    "summary_stats": initial_data.describe().to_dict(),
                })
            else:
                context["initial_data"] = str(initial_data)
        return context

    def create_plan(self, objective: str, initial_data: Any | None = None) -> Plan:
        """Create an execution plan for a given objective and optional data."""
        context = self._prepare_context(objective, initial_data)
        logger.info("Generating execution plan with LLM…")
        llm_plan = self.plan_generator.generate_plan(objective, context)
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tasks: List[Task] = []
        for i, task_plan in enumerate(llm_plan.tasks):
            task = Task(
                id=f"{plan_id}_task{i+1}",
                name=task_plan.name,
                description=task_plan.description,
            )
            # convert subtasks
            subtasks = []
            for j, subtask_plan in enumerate(task_plan.subtasks):
                subtask = SubTask(
                    id=f"{task.id}_sub{j+1}",
                    name=subtask_plan.name,
                    description=subtask_plan.description,
                    reasoning=subtask_plan.reasoning,
                )
                subtasks.append(subtask)
            task.subtasks = subtasks
            tasks.append(task)
        plan = Plan(
            id=plan_id,
            objective=llm_plan.objective,
            tasks=tasks,
            reasoning=llm_plan.reasoning,
            success_criteria=llm_plan.success_criteria,
        )
        self.plans.append(plan)
        self.current_plan = plan
        # store initial data in sandbox
        if initial_data is not None:
            self.sandbox.set_variable("initial_data", initial_data)
            if isinstance(initial_data, pd.DataFrame):
                self.sandbox.set_variable("df", initial_data)
        return plan

    def _execute_subtask_sync(self, subtask: SubTask) -> None:
        """Execute a single subtask synchronously."""
        subtask.status = TaskStatus.RUNNING
        logger.info(f"Executing subtask: {subtask.name}")
        try:
            context = self.sandbox.get_context()
            code_gen = self.code_generator.generate_code(subtask, context)
            subtask.code = code_gen.code
            logger.info(f"Code explanation: {code_gen.explanation}")
            result, stdout, error = self.sandbox.execute(code_gen.code)
            if error:
                subtask.status = TaskStatus.FAILED
                subtask.error = error
                logger.error(f"Execution failed: {error}")
            else:
                subtask.status = TaskStatus.COMPLETED
                subtask.result = {
                    "output": stdout,
                    "value": result,
                    "new_variables": list(self.sandbox.get_context().keys()),
                }
                subtask.completed_at = datetime.now()
                subtask.llm_analysis = self.reasoning_engine.analyze_results(
                    subtask, stdout, self.sandbox.get_context()
                )
                logger.info(f"Analysis: {subtask.llm_analysis[:200]}…")
        except Exception as e:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            logger.error(f"Unexpected error while executing subtask {subtask.name}: {e}")

    async def _execute_subtask_async(self, subtask: SubTask) -> None:
        """Execute a single subtask asynchronously."""
        self._execute_subtask_sync(subtask)
        await asyncio.sleep(0.1)

    def execute_plan_sync(self) -> None:
        """Execute the current plan synchronously."""
        if not self.current_plan:
            raise ValueError("No plan has been created.")
        self.current_plan.status = TaskStatus.RUNNING
        logger.info(f"Executing plan: {self.current_plan.objective}")
        try:
            for task in self.current_plan.tasks:
                task.status = TaskStatus.RUNNING
                logger.info("=" * 60)
                logger.info(f"Starting task: {task.name}")
                logger.info("=" * 60)
                for subtask in task.subtasks:
                    self._execute_subtask_sync(subtask)
                    # readability delay
                    import time
                    time.sleep(0.3)
                if all(st.status == TaskStatus.COMPLETED for st in task.subtasks):
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                else:
                    task.status = TaskStatus.FAILED
                    break
            if all(t.status == TaskStatus.COMPLETED for t in self.current_plan.tasks):
                self.current_plan.status = TaskStatus.COMPLETED
                self.current_plan.completed_at = datetime.now()
                results_summary = self._get_results_summary()
                final_recs = self.reasoning_engine.generate_final_recommendations(
                    self.current_plan, results_summary
                )
                self.execution_history.append({
                    "plan_id": self.current_plan.id,
                    "objective": self.current_plan.objective,
                    "status": "completed",
                    "recommendations": final_recs,
                    "timestamp": datetime.now().isoformat(),
                })
                logger.info("Final recommendations:\n" + final_recs)
            else:
                self.current_plan.status = TaskStatus.FAILED
        except Exception as e:
            self.current_plan.status = TaskStatus.FAILED
            logger.error(f"Plan execution failed: {e}")
            raise

    async def execute_plan(self) -> None:
        """Execute the current plan asynchronously."""
        if not self.current_plan:
            raise ValueError("No plan has been created.")
        self.current_plan.status = TaskStatus.RUNNING
        logger.info(f"Executing plan: {self.current_plan.objective}")
        try:
            for task in self.current_plan.tasks:
                task.status = TaskStatus.RUNNING
                logger.info("=" * 60)
                logger.info(f"Starting task: {task.name}")
                logger.info("=" * 60)
                for subtask in task.subtasks:
                    await self._execute_subtask_async(subtask)
                    await asyncio.sleep(0.3)
                if all(st.status == TaskStatus.COMPLETED for st in task.subtasks):
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                else:
                    task.status = TaskStatus.FAILED
                    break
            if all(t.status == TaskStatus.COMPLETED for t in self.current_plan.tasks):
                self.current_plan.status = TaskStatus.COMPLETED
                self.current_plan.completed_at = datetime.now()
                results_summary = self._get_results_summary()
                final_recs = self.reasoning_engine.generate_final_recommendations(
                    self.current_plan, results_summary
                )
                self.execution_history.append({
                    "plan_id": self.current_plan.id,
                    "objective": self.current_plan.objective,
                    "status": "completed",
                    "recommendations": final_recs,
                    "timestamp": datetime.now().isoformat(),
                })
                logger.info("Final recommendations:\n" + final_recs)
            else:
                self.current_plan.status = TaskStatus.FAILED
        except Exception as e:
            self.current_plan.status = TaskStatus.FAILED
            logger.error(f"Plan execution failed: {e}")
            raise

    def _get_results_summary(self) -> Dict[str, Any]:
        """Aggregate all subtask analyses into a nested dict."""
        summary: Dict[str, Any] = {}
        if not self.current_plan:
            return summary
        for task in self.current_plan.tasks:
            task_results: Dict[str, Any] = {}
            for subtask in task.subtasks:
                if subtask.status == TaskStatus.COMPLETED:
                    task_results[subtask.name] = {
                        "analysis": subtask.llm_analysis,
                        "output_preview": subtask.result.get("output", "")[:300]
                        if subtask.result
                        else "",
                    }
            summary[task.name] = task_results
        return summary

    def get_execution_report(self) -> str:
        """Produce a human readable report of the plan execution."""
        if not self.current_plan:
            return "No plan has been executed."
        plan = self.current_plan
        report_lines = [
            "# Data Science Agent Execution Report\n",
            "## Objective\n",
            f"{plan.objective}\n",
            "\n## Execution Status\n",
            f"- Plan ID: {plan.id}\n",
            f"- Status: {plan.status.value}\n",
            f"- Created: {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- Completed: {plan.completed_at.strftime('%Y-%m-%d %H:%M:%S') if plan.completed_at else 'N/A'}\n",
            "\n## Success Criteria\n",
            f"{plan.success_criteria}\n",
            "\n## Strategy\n",
            f"{plan.reasoning}\n",
            "\n## Task Execution Summary\n",
        ]
        for task in plan.tasks:
            report_lines.append(f"\n### {task.name}\n")
            report_lines.append(f"Status: {task.status.value}\n\n")
            for subtask in task.subtasks:
                report_lines.append(f"#### {subtask.name}\n")
                report_lines.append(f"- Status: {subtask.status.value}\n")
                if subtask.llm_analysis:
                    report_lines.append(f"- Analysis: {subtask.llm_analysis}\n")
                if subtask.error:
                    report_lines.append(f"- Error: {subtask.error}\n")
                report_lines.append("\n")
        # Append final recommendations if available
        if self.execution_history:
            latest = self.execution_history[-1]
            recs = latest.get("recommendations")
            if recs:
                report_lines.append("\n## Final Recommendations\n")
                report_lines.append(recs + "\n")
        return "".join(report_lines)

    def save_results(self, filepath: str) -> None:
        """Persist plan execution details and context to a JSON file."""
        if not self.current_plan:
            raise ValueError("No plan has been executed.")
        results = {
            "plan": {
                "id": self.current_plan.id,
                "objective": self.current_plan.objective,
                "status": self.current_plan.status.value,
                "reasoning": self.current_plan.reasoning,
                "success_criteria": self.current_plan.success_criteria,
            },
            "execution_history": self.execution_history,
            "final_context": {k: str(v) for k, v in self.sandbox.get_context().items()},
            "report": self.get_execution_report(),
        }
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")