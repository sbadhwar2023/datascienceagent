"""Reasoning and insight generation for the Data Science Agent.

This module defines the :class:`ReasoningEngine`, responsible for
calling a language model to analyse intermediate results and to
produce final recommendations. Two methods are provided:

``analyze_results``
    Given the output of a subtask and the current context, generates
    insights, implications and suggestions for further work.

``generate_final_recommendations``
    Summarises all collected results and delivers an executive summary
    with actionable recommendations and observations on data quality.
"""

from __future__ import annotations

import json
from typing import Dict, Any

from .plan import Plan, SubTask
from .llm_interface import BaseLLM


class ReasoningEngine:
    """Encapsulates calls to the LLM for analysis and recommendations."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def analyze_results(self, subtask: SubTask, output: str, context: Dict[str, Any]) -> str:
        """Generate analysis of subtask results using the LLM.

        :param subtask: the subtask for which results are being analysed
        :param output: captured standard output from executing the subtask
        :param context: current variables in the sandbox
        :returns: a string containing key findings, implications and next steps
        """
        prompt = (
            "As an expert data scientist, analyse the following results from a subtask:\n\n"
            f"Subtask: {subtask.name}\n"
            f"Description: {subtask.description}\n\n"
            "Output (truncated to 1500 chars):\n"
            f"{output[:1500]}\n\n"
            f"Context variable names: {list(context.keys())}\n\n"
            "Provide:\n"
            "1. Key findings and insights\n"
            "2. Implications for the overall objective\n"
            "3. Specific suggestions for next steps\n"
            "4. Any concerns, limitations, or data quality issues\n"
            "5. How these results connect to previous findings (if any)\n"
            "Be concise but thorough in your analysis."
        )
        messages = [
            {"role": "system", "content": "You are an expert data scientist providing insightful analysis."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm(messages)
        return response.content

    def generate_final_recommendations(self, plan: Plan, results: Dict[str, Any]) -> str:
        """Produce final recommendations once all tasks have completed.

        :param plan: the executed plan, including objective and success criteria
        :param results: a nested dict containing analyses for each subtask
        :returns: a string with summary, recommendations, expected outcomes and limitations
        """
        prompt = (
            "As an expert data scientist, provide final recommendations based on the completed analysis:\n\n"
            f"Objective: {plan.objective}\n"
            f"Success Criteria: {plan.success_criteria}\n\n"
            "Summary of findings:\n"
            f"{json.dumps(results, indent=2, default=str)[:2500]}\n\n"
            "Please provide:\n"
            "1. Executive summary (2-3 sentences) of the most critical findings\n"
            "2. Top 3-5 specific, actionable recommendations with clear rationale\n"
            "3. Expected outcomes if recommendations are implemented\n"
            "4. Potential risks or considerations to monitor\n"
            "5. Suggested next experiments or analyses to further optimise results\n"
            "6. Data quality observations and any limitations of the analysis\n"
            "Focus on practical, implementable suggestions backed by the data analysis."
        )
        messages = [
            {"role": "system", "content": "You are an expert data scientist providing strategic recommendations."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm(messages)
        return response.content