"""The agent's tools.

The brief requires **exactly two** tools; a third is added as the bonus.

1. ``calculator`` — evaluates an arithmetic expression safely. The
   expression is parsed with :mod:`ast` and only numbers and arithmetic
   operators are allowed — ``eval()`` is never used.
2. ``file_writer`` — writes text to a file inside a sandboxed output
   folder. File names are reduced to their base name, so a goal can
   never trick the agent into writing outside the sandbox.
3. ``web_search`` *(bonus)* — a quick lookup through the free DuckDuckGo
   Instant Answers API (no key required).
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Callable

import requests


class ToolError(Exception):
    """Raised when a tool cannot complete its task."""


# ----------------------------------------------------------------------
# Tool 1: calculator
# ----------------------------------------------------------------------
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _evaluate(node: ast.AST) -> float:
    """Recursively evaluate an AST node made only of numbers/operators."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError("Only numbers are allowed in expressions.")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("Division by zero is not allowed.")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ToolError(
        "Expression contains something that is not plain arithmetic."
    )


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. ``"240 * 0.15"`` → ``"36"``.

    Raises:
        ToolError: If the expression is not safe, plain arithmetic.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression '{expression}': {exc.msg}") from exc
    result = _evaluate(tree)
    if result == int(result) and abs(result) < 1e15:
        return str(int(result))
    return f"{result:.10g}"


# ----------------------------------------------------------------------
# Tool 2: file writer (sandboxed)
# ----------------------------------------------------------------------
class FileWriter:
    """Write text files inside one sandbox directory.

    Args:
        output_dir: The only directory this tool may write to.
    """

    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = Path(output_dir)

    def __call__(self, filename: str, content: str) -> str:
        """Write ``content`` to ``filename`` inside the sandbox.

        The file name is reduced to its base name — directory parts are
        stripped, so ``../../etc/passwd`` becomes ``passwd`` inside the
        sandbox rather than escaping it.

        Raises:
            ToolError: If the file name is empty after sanitising.
        """
        safe_name = Path(filename).name
        if not safe_name:
            raise ToolError("A file name is required.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / safe_name
        target.write_text(content, encoding="utf-8")
        return f"Saved {len(content)} characters to {target}"


# ----------------------------------------------------------------------
# Tool 3 (bonus): web search
# ----------------------------------------------------------------------
def web_search(query: str) -> str:
    """Look up a short factual summary via DuckDuckGo Instant Answers.

    Raises:
        ToolError: On network failure.
    """
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise ToolError(f"Web search failed: {exc}") from exc

    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        source = data.get("AbstractSource", "DuckDuckGo")
        return f"{abstract} (source: {source})"
    topics = data.get("RelatedTopics") or []
    for topic in topics:
        text = (topic.get("Text") or "").strip()
        if text:
            return text
    return f"No instant answer found for '{query}'."


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def build_tools(output_dir: str | Path = "output") -> dict[str, Callable[..., str]]:
    """Return the agent's tool registry: two required tools + the bonus."""
    return {
        "calculator": calculator,
        "file_writer": FileWriter(output_dir),
        "web_search": web_search,
    }


#: JSON-schema descriptions of the tools, in OpenAI function-calling form.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a plain arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '240 * 0.15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_writer",
            "description": "Save text content to a named file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Look up a short factual summary on the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]
