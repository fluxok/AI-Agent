from __future__ import annotations

from typing import Tuple

from langgraph.graph import StateGraph, START

from .nodes import make_classic_riddle, make_other_task, make_sequence_solver
from .prompt import PromptRunner
from .state import AgentState


def build_graph(model: str) -> Tuple[PromptRunner, any]:
    runner = PromptRunner(model)
    graph = StateGraph(AgentState)
    graph.add_node("sequence_solver", make_sequence_solver(runner))
    graph.add_node("classic_riddle", make_classic_riddle(runner))
    graph.add_node("other_task", make_other_task(runner))
    graph.add_conditional_edges(
        START,
        route_topic,
        {
            "sequence_solver": "sequence_solver",
            "classic_riddle": "classic_riddle",
            "other_task": "other_task",
        },
    )
    return runner, graph.compile()


def route_topic(state: AgentState) -> str:
    raw_topic = state.get("topic") or ""
    topic = str(raw_topic).lower()
    if "sequence" in topic:
        return "sequence_solver"
    if any(key in topic for key in ("riddle", "puzzle")):
        return "classic_riddle"
    return "other_task"
