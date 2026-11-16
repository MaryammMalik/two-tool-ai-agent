"""Command-line interface for the two-tool AI agent.

Usage:
    python -m agent "What's 15% of 240, and save the answer to a file called result.txt?"
    python -m agent "Search for the capital of France" --log-file run.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core import DEFAULT_MAX_STEPS, Agent
from .llm import get_planner
from .tools import build_tools


def setup_logging(log_file: str | None) -> logging.Logger:
    """Configure the agent logger: console, plus a file when requested.

    Only tool calls, tool results, and run start/end lines are ever
    logged — never the planner's internal reasoning.
    """
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Give a natural-language goal to a two-tool AI agent.",
    )
    parser.add_argument("goal", nargs="+", help="The goal, in plain words")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum agent steps. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder the file_writer tool may write to. Default: output/",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write the tool-call log to this file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    goal = " ".join(args.goal).strip()
    if not goal:
        print("Error: a goal is required.", file=sys.stderr)
        return 2

    logger = setup_logging(args.log_file)
    planner = get_planner()
    agent = Agent(
        planner=planner,
        tools=build_tools(Path(args.output_dir)),
        max_steps=args.max_steps,
        logger=logger,
    )
    result = agent.run(goal)

    print(f"\nAgent: {result.final_answer}")
    print(
        f"Finished in {result.steps_used} step(s), "
        f"{result.llm_calls} LLM call(s), "
        f"{len(result.tool_calls)} tool call(s)."
    )
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
