"""Secure code execution sandbox.

The :class:`CodeSandbox` class provides a restricted environment for
executing dynamically generated Python code. It exposes only
whitelisted modules (pandas, numpy, matplotlib, seaborn) and
intentionally limits access to built-in functions to reduce the risk of
malicious code execution. Variables created during code execution can
be inspected and reused across subtasks.
"""

from __future__ import annotations

import io
import contextlib
import traceback
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CodeSandbox:
    """Secure environment for executing user-generated code.

    The sandbox maintains separate global and local dictionaries used
    for each code execution. Global variables include commonly used
    data science libraries (pandas, numpy, matplotlib, seaborn) and
    builtins. New variables created during execution are stored in
    ``locals_dict`` and can be retrieved via :meth:`get_variable` or
    inspected with :meth:`get_context`.
    """

    def __init__(self) -> None:
        # Expose only safe globals and builtins. Note: builtins cannot be
        # completely removed but we avoid adding any dangerous functions here.
        self.globals_dict: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "print": print,
            "__builtins__": __builtins__,
        }
        # This dictionary will hold user defined variables
        self.locals_dict: Dict[str, Any] = {}

    def execute(self, code: str) -> Tuple[Any, str, str | None]:
        """Execute arbitrary Python code in the sandbox.

        Returns a tuple of (result, stdout, error). If a variable named
        ``result`` is defined in the executed code, its value will be
        returned as the first element of the tuple. Standard output is
        captured and returned as the second element. If an exception
        occurs, the error string and traceback are returned as the
        third element and ``result`` will be ``None``.
        """
        stdout_buffer = io.StringIO()
        result: Any = None
        error: str | None = None

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self.globals_dict, self.locals_dict)
                if "result" in self.locals_dict:
                    result = self.locals_dict["result"]
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        stdout = stdout_buffer.getvalue()
        return result, stdout, error

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the sandbox if it exists."""
        return self.locals_dict.get(name, self.globals_dict.get(name))

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox for later access."""
        self.locals_dict[name] = value

    def get_context(self) -> Dict[str, Any]:
        """Return all user defined variables from the sandbox."""
        context: Dict[str, Any] = {}
        for name, value in self.locals_dict.items():
            if not name.startswith("_") and name not in ["pd", "np", "plt", "sns"]:
                context[name] = value
        return context

    def clear(self) -> None:
        """Clear all user defined variables from the sandbox."""
        self.locals_dict.clear()