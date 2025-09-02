"""Code generation using language models.

This module defines the :class:`CodeGenerator` class which wraps a
language model to produce executable Python code for a specific
subtask. The code is returned alongside an explanation, the expected
output and a list of required context variables. A Pydantic output
parser ensures that the model response conforms to the expected JSON
schema.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage, SystemMessage

from .plan import SubTask, CodeGeneration
from .llm_interface import BaseLLM

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generate Python code for a given subtask using an LLM."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.code_parser = PydanticOutputParser(pydantic_object=CodeGeneration)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.code_parser, llm=self.llm
        )

    def generate_code(self, subtask: SubTask, context: Dict[str, Any]) -> CodeGeneration:
        """Generate Python code for the supplied subtask.

        :param subtask: The subtask definition containing name and description.
        :param context: Current execution context (variables) available to the code.
        :returns: A :class:`CodeGeneration` instance containing code and metadata.
        """
        context_vars_description = "\n".join([
            f"- {k}: {type(v).__name__}" for k, v in context.items()
        ]) if context else "None"

        system_prompt = (
            "You are an expert data scientist writing Python code. "
            "Generate clean, efficient code that accomplishes the given subtask. "
            "Use pandas, numpy, matplotlib, and seaborn as needed. "
            "Always store important results in variables that can be accessed later. "
            "Include helpful print statements to show progress and results.\n\n"
            "Available context variables:\n{context_vars}\n\n"
            "Return a valid JSON response with the following fields:\n"
            "{format_instructions}"
        )
        user_prompt = (
            f"Subtask: {subtask.name}\n"
            f"Description: {subtask.description}\n\n"
            "Current context variable names:\n"
            f"{json.dumps(list(context.keys()), indent=2)}\n\n"
            "Generate Python code that:\n"
            "1. Accomplishes this subtask\n"
            "2. Stores results in meaningful variables\n"
            "3. Prints informative output\n"
            "4. Handles potential errors gracefully\n\n"
            "Format your response as JSON with code, explanation, expected_output, and required_context fields."
        )
        messages = [
            {"role": "system", "content": system_prompt.format(
                context_vars=context_vars_description,
                format_instructions=self.code_parser.get_format_instructions()
            )},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm(messages)
        try:
            code_gen = self.code_parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Initial code parsing failed: {e}. Attempting to fix output...")
            code_gen = self.fixing_parser.parse(response.content)
        return code_gen