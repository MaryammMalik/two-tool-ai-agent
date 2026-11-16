"""Planners: the "brain" that decides the agent's next action.

A planner looks at the goal and the history of tool results so far, and
returns exactly one action: call a tool, or finish with an answer.

* :class:`OpenAIPlanner` — a real LLM (OpenAI, function calling) decides,
  as in Module 8. Every decision is one LLM call, and the planner counts
  them (Task 9 bonus).
* :class:`ScriptedPlanner` — a deterministic, rule-based stand-in used
  when no API key is configured, so the agent loop, tools, logging and
  step limit can be run and tested end to end offline. It is always
  labelled as the offline planner in the run output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .tools import TOOL_SCHEMAS

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a task agent. You can call tools to complete the user's "
    "goal. Use the calculator for arithmetic, the file_writer to save "
    "results to a file, and web_search for factual lookups. When the "
    "goal is fully complete, reply with a short plain-text summary "
    "instead of calling another tool."
)


@dataclass
class ToolCall:
    """The planner decided to call a tool."""

    name: str
    arguments: dict


@dataclass
class Finish:
    """The planner decided the goal is complete."""

    answer: str


Action = ToolCall | Finish


class Planner(Protocol):
    """Anything that can decide the agent's next action."""

    name: str
    calls: int

    def decide(self, goal: str, history: list[dict]) -> Action:
        """Return the next action given the goal and tool history."""
        ...


class OpenAIPlanner:
    """Decide actions with the OpenAI API's function calling.

    Args:
        api_key: OpenAI API key (read from ``OPENAI_API_KEY`` if omitted).
        model: Chat model name; override with ``OPENAI_MODEL``.
    """

    name = "OpenAI"

    def __init__(
        self, api_key: str | None = None, model: str | None = None
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.calls = 0

    def decide(self, goal: str, history: list[dict]) -> Action:
        """Ask the LLM for the next action (counts as one LLM call)."""
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        for i, item in enumerate(history):
            call_id = f"call_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item["tool"],
                                "arguments": json.dumps(item["arguments"]),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": item["result"],
                }
            )

        self.calls += 1
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            return ToolCall(name=call.function.name, arguments=arguments)
        return Finish(answer=(message.content or "").strip())


_PERCENTAGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_SAVE_TARGET = re.compile(
    r"(?:save|write|store).*?(?:called|named|to|as)\s+"
    r"(?:a file called\s+)?['\"]?([\w.\-]+\.\w+)['\"]?",
    re.IGNORECASE,
)
_ARITHMETIC = re.compile(r"\d\s*[-+*/]\s*\d")


class ScriptedPlanner:
    """Deterministic offline planner (no API key needed).

    Understands the goal shapes this agent is built for:

    * "What is X% of Y, and save the answer to a file called Z?"
      (the worked example from the task brief)
    * a bare arithmetic goal, e.g. "Calculate 12 * 8 + 5"
    * a lookup goal, e.g. "Search for the capital of France"

    Anything else finishes immediately with an explanation rather than
    looping. ``calls`` counts decisions, mirroring the LLM-call counter.
    """

    name = "offline scripted planner (no API key set)"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, goal: str, history: list[dict]) -> Action:
        """Return the next scripted action for ``goal``."""
        self.calls += 1
        done = {item["tool"] for item in history}
        results = {item["tool"]: item["result"] for item in history}

        # A finished file write completes the example workflow.
        if "file_writer" in done:
            saved = results["file_writer"]
            calculated = results.get("calculator", "")
            filename = next(
                item["arguments"].get("filename", "a file")
                for item in history
                if item["tool"] == "file_writer"
            )
            if calculated:
                return Finish(
                    answer=(
                        f"Calculated {calculated} and saved it "
                        f"to {filename}."
                    )
                )
            return Finish(answer=f"Done. {saved}")

        # A search result completes a lookup goal.
        if "web_search" in done:
            return Finish(answer=f"Search result: {results['web_search']}")

        # Step 1 for a percentage goal: calculate.
        percentage = _PERCENTAGE.search(goal)
        if percentage and "calculator" not in done:
            percent, base = percentage.groups()
            return ToolCall(
                name="calculator",
                arguments={"expression": f"{base} * {percent} / 100"},
            )

        # A calculation exists and the goal asks to save it: write it.
        if "calculator" in done:
            target = _SAVE_TARGET.search(goal)
            if target:
                return ToolCall(
                    name="file_writer",
                    arguments={
                        "filename": target.group(1),
                        "content": results["calculator"],
                    },
                )
            return Finish(answer=f"The result is {results['calculator']}.")

        # A bare arithmetic goal: calculate it directly.
        arithmetic = _ARITHMETIC.search(goal)
        if arithmetic and "calculator" not in done:
            expression = goal[arithmetic.start():]
            expression = re.split(r"(?:,|\band\b|\bthen\b)", expression)[0]
            return ToolCall(
                name="calculator",
                arguments={"expression": expression.strip()},
            )

        # A lookup goal: use the bonus search tool.
        lowered = goal.lower()
        if "search" in lowered or lowered.startswith(("what is", "who is")):
            query = re.sub(
                r"^(search for|search|look up)\s+", "", goal, flags=re.IGNORECASE
            ).strip().rstrip("?")
            return ToolCall(name="web_search", arguments={"query": query})

        return Finish(
            answer=(
                "I could not break that goal into tool calls with the "
                "offline planner. Set OPENAI_API_KEY to use the full "
                "LLM planner."
            )
        )


def get_planner() -> Planner:
    """Return the OpenAI planner when a key exists, else the scripted one."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return OpenAIPlanner()
    return ScriptedPlanner()
