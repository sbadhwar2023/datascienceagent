"""Plan generation using language models.

This module defines the :class:`PlanGenerator` class which wraps a
language model to produce detailed execution plans. It takes a user
objective and an optional context dictionary and returns a
structured :class:`~data_science_agent.plan.ExecutionPlan` by
leveraging Pydantic output parsers.

The default prompt instructs the model to organise work into tasks
and subtasks, providing reasoning and success criteria. You can
subclass :class:`PlanGenerator` to customise the prompts or the
behaviour for different domains.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, List

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage, SystemMessage

from .plan import ExecutionPlan
from .llm_interface import BaseLLM

logger = logging.getLogger(__name__)


class PlanGenerator:
    """Generate execution plans using a language model.

    The generator uses a Pydantic output parser to convert the raw
    model output into a strongly typed :class:`ExecutionPlan`. If the
    initial parse fails due to formatting issues the output fixing
    parser is applied as a fallback.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.plan_parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.plan_parser, llm=self.llm
        )

    def generate_plan(self, objective: str, context: Dict[str, Any]) -> ExecutionPlan:
        """Generate a detailed execution plan for the given objective.

        :param objective: high level description of what the agent
            should accomplish
        :param context: dictionary containing metadata about the
            problem, e.g. dataset information
        :returns: a validated :class:`ExecutionPlan`
        """
        system_prompt = (
            "You are an expert data scientist creating detailed execution plans. "
            "Given an objective and context, create a comprehensive plan with tasks and subtasks. "
            "Each task should have clear subtasks that build upon each other. "
            "Focus on systematic analysis, insights extraction, and actionable recommendations. "
            "Return a valid JSON structure that matches this format:\n"
            "{format_instructions}"
        )
        user_prompt = (
            f"Objective: {objective}\n\n"
            f"Context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Create a detailed execution plan with:\n"
            "1. Clear tasks that progress logically\n"
            "2. Specific subtasks with expected outputs\n"
            "3. Reasoning for each step\n"
            "4. Success criteria for the overall objective\n\n"
            "Remember to format your response as valid JSON matching the required structure."
        )
        messages = [
            {"role": "system", "content": system_prompt.format(
                format_instructions=self.plan_parser.get_format_instructions()
            )},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm(messages)
        try:
            plan = self.plan_parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Initial plan parsing failed: {e}. Attempting to fix output...")
            plan = self.fixing_parser.parse(response.content)
        return plan