"""Tests for the two-tool AI agent.

The behaviours under test are the ones the brief calls out: exactly two
core tools (plus the bonus third), a **maximum of 5 steps**, **logging of
tool calls and results only**, and the bonus **LLM-call counter**.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agent.core import Agent
from agent.llm import Finish, ScriptedPlanner, ToolCall
from agent.tools import FileWriter, ToolError, build_tools, calculator, web_search

EXAMPLE_GOAL = (
    "What's 15% of 240, and save the answer to a file called result.txt?"
)


class ScriptedTestPlanner:
    """Returns a fixed queue of actions, counting its own calls."""

    name = "test planner"

    def __init__(self, actions: list) -> None:
        self._actions = list(actions)
        self.calls = 0

    def decide(self, goal: str, history: list[dict]):
        self.calls += 1
        if self._actions:
            return self._actions.pop(0)
        return Finish(answer="done")


def make_agent(tmp_path: Path, planner, max_steps: int = 5) -> Agent:
    return Agent(
        planner=planner,
        tools=build_tools(tmp_path / "output"),
        max_steps=max_steps,
        logger=logging.getLogger("agent-test"),
    )


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
def test_calculator_arithmetic() -> None:
    assert calculator("240 * 0.15") == "36"
    assert calculator("12 * 8 + 5") == "101"
    assert calculator("(10 + 5) / 3") == "5"
    assert calculator("-7 + 2") == "-5"


def test_calculator_rejects_code_and_division_by_zero() -> None:
    with pytest.raises(ToolError):
        calculator("__import__('os').system('echo hi')")
    with pytest.raises(ToolError):
        calculator("1 / 0")
    with pytest.raises(ToolError):
        calculator("not maths")


def test_file_writer_stays_in_sandbox(tmp_path: Path) -> None:
    writer = FileWriter(tmp_path)
    message = writer("../../evil.txt", "36")
    assert (tmp_path / "evil.txt").read_text(encoding="utf-8") == "36"
    assert "evil.txt" in message
    assert not (tmp_path.parent / "evil.txt").exists()


def test_web_search_uses_duckduckgo(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.json.return_value = {
        "AbstractText": "Paris is the capital of France.",
        "AbstractSource": "Wikipedia",
    }
    response.raise_for_status.return_value = None
    with patch("agent.tools.requests.get", return_value=response):
        result = web_search("capital of France")
    assert "Paris" in result


def test_exactly_two_required_tools_plus_bonus() -> None:
    tools = build_tools("output")
    assert set(tools) == {"calculator", "file_writer", "web_search"}


# ----------------------------------------------------------------------
# The brief's example workflow, end to end (offline planner)
# ----------------------------------------------------------------------
def test_example_goal_end_to_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = make_agent(tmp_path, ScriptedPlanner())
    with caplog.at_level(logging.INFO, logger="agent-test"):
        result = agent.run(EXAMPLE_GOAL)

    assert result.completed is True
    assert "36" in result.final_answer
    assert (tmp_path / "output" / "result.txt").read_text(
        encoding="utf-8"
    ) == "36"
    assert [c.tool for c in result.tool_calls] == [
        "calculator",
        "file_writer",
    ]
    # Bonus: the LLM-call counter — 2 tool decisions + 1 finish decision.
    assert result.llm_calls == 3
    # Logging: both tool calls and their results are in the log.
    log_text = "\n".join(r.message for r in caplog.records)
    assert "TOOL_CALL" in log_text and "calculator" in log_text
    assert "TOOL_RESULT" in log_text and "file_writer" in log_text


# ----------------------------------------------------------------------
# Loop guarantees
# ----------------------------------------------------------------------
def test_step_limit_stops_a_looping_agent(tmp_path: Path) -> None:
    planner = ScriptedTestPlanner(
        [ToolCall("calculator", {"expression": "1 + 1"})] * 20
    )
    agent = make_agent(tmp_path, planner, max_steps=5)
    result = agent.run("Keep calculating forever")
    assert result.completed is False
    assert result.steps_used == 5
    assert result.llm_calls == 5
    assert len(result.tool_calls) == 5
    assert "step limit" in result.final_answer


def test_tool_error_becomes_result_not_crash(tmp_path: Path) -> None:
    planner = ScriptedTestPlanner(
        [
            ToolCall("calculator", {"expression": "1 / 0"}),
            Finish(answer="handled"),
        ]
    )
    result = make_agent(tmp_path, planner).run("Divide by zero")
    assert result.completed is True
    assert "Error" in result.tool_calls[0].result


def test_unknown_tool_is_reported(tmp_path: Path) -> None:
    planner = ScriptedTestPlanner(
        [ToolCall("teleporter", {}), Finish(answer="ok")]
    )
    result = make_agent(tmp_path, planner).run("Beam me up")
    assert "unknown tool" in result.tool_calls[0].result


def test_empty_goal_rejected(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, ScriptedPlanner())
    with pytest.raises(ValueError):
        agent.run("   ")


def test_max_steps_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Agent(
            planner=ScriptedPlanner(),
            tools=build_tools(tmp_path),
            max_steps=0,
        )
