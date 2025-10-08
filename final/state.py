from __future__ import annotations

from typing import List, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    query: str
    topic: str
    options: List[str]
    reasoning: List[str]
    answer_option: int
    confidence: float


def make_agent_state(row: pd.Series) -> AgentState:
    options = [
        value.strip()
        for idx in range(1, 5)
        for value in [row.get(f"answer_option_{idx}")]
        if isinstance(value, str) and value.strip()
    ]
    return AgentState(
        query=row.get("problem_statement", ""),
        topic=row.get("topic", ""),
        options=options,
        reasoning=[],
        answer_option=5,
        confidence=0.0,
    )
