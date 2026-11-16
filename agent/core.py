"""The agent loop.

Guarantees from the brief, enforced here:

* **Limited execution loop** — at most ``max_steps`` steps (default 5),
  then the agent stops, finished or not.
* **Logging of actions only** — every tool call and its result is logged
  (tool name, inputs, output). The planner's internal reasoning is never
  logged, because it is never even requested.
* **LLM-call counter** (bonus) — the number of planner decisions (LLM
  calls) is tracked and reported for every run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from .llm import Finish, Planner, ToolCall
from .tools import ToolError

DEFAULT_MAX_STEPS = 5


@dataclass
class ToolCallRecord:
    """One executed tool call and its result."""

    step: int
    tool: str
    arguments: dict
    result: str


@dataclass
class RunResult:
    """The outcome of one agent run."""

    goal: str
    final_answer: str
    steps_used: int
    llm_calls: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    completed: bool = False


class Agent:
    """Run a goal through a planner + tools loop with a hard step cap.

    Args:
        planner: Decides each next action (an LLM, or the offline
            scripted planner).
        tools: Mapping of tool name to callable.
        max_steps: Maximum number of planner decisions before the run
            is stopped. Must be at least 1.
        logger: Logger for tool-call records; defaults to ``"agent"``.
    """

    def __init__(
        self,
        planner: Planner,
        tools: dict[str, Callable[..., str]],
        max_steps: int = DEFAULT_MAX_STEPS,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.planner = planner
        self.tools = tools
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger("agent")

    def run(self, goal: str) -> RunResult:
        """Execute ``goal`` and return a :class:`RunResult`.

        Tool failures do not crash the loop: the error becomes the
        tool's result, is logged, and the planner can react to it.
        """
        goal = goal.strip()
        if not goal:
            raise ValueError("Goal must not be empty.")

        history: list[dict] = []
        records: list[ToolCallRecord] = []
        self.logger.info(
            "RUN_START goal=%r planner=%s max_steps=%d",
            goal,
            self.planner.name,
            self.max_steps,
        )

        for step in range(1, self.max_steps + 1):
            action = self.planner.decide(goal, history)

            if isinstance(action, Finish):
                self.logger.info(
                    "RUN_END steps=%d llm_calls=%d tool_calls=%d "
                    "answer=%r",
                    step - 1,
                    self.planner.calls,
                    len(records),
                    action.answer,
                )
                return RunResult(
                    goal=goal,
                    final_answer=action.answer,
                    steps_used=step - 1,
                    llm_calls=self.planner.calls,
                    tool_calls=records,
                    completed=True,
                )

            if isinstance(action, ToolCall):
                result = self._execute(action, step)
                records.append(
                    ToolCallRecord(
                        step=step,
                        tool=action.name,
                        arguments=action.arguments,
                        result=result,
                    )
                )
                history.append(
                    {
                        "tool": action.name,
                        "arguments": action.arguments,
                        "result": result,
                    }
                )

        answer = (
            f"Stopped after reaching the {self.max_steps}-step limit "
            "before the goal was complete."
        )
        self.logger.info(
            "RUN_END steps=%d llm_calls=%d tool_calls=%d answer=%r",
            self.max_steps,
            self.planner.calls,
            len(records),
            answer,
        )
        return RunResult(
            goal=goal,
            final_answer=answer,
            steps_used=self.max_steps,
            llm_calls=self.planner.calls,
            tool_calls=records,
            completed=False,
        )

    def _execute(self, action: ToolCall, step: int) -> str:
        """Run one tool call, logging the call and its result."""
        self.logger.info(
            "TOOL_CALL step=%d tool=%s arguments=%s",
            step,
            action.name,
            json.dumps(action.arguments, default=str),
        )
        tool = self.tools.get(action.name)
        if tool is None:
            result = f"Error: unknown tool '{action.name}'."
        else:
            try:
                result = str(tool(**action.arguments))
            except ToolError as exc:
                result = f"Error: {exc}"
            except TypeError as exc:
                result = f"Error: bad arguments for '{action.name}': {exc}"
            except Exception as exc:  # a tool must never crash the loop
                result = f"Error: {type(exc).__name__}: {exc}"
        self.logger.info(
            "TOOL_RESULT step=%d tool=%s result=%r", step, action.name, result
        )
        return result
