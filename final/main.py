from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from graph import build_graph
from state import AgentState, make_agent_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mini reasoning pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="CSV file with problems and options.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for predictions.")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model to query.")
    return parser.parse_args()


def run_pipeline(input_csv: Path, output_csv: Path | None, model: str) -> pd.DataFrame:
    runner, compiled_graph = build_graph(model)
    frame = pd.read_csv(input_csv)
    rows: List[Dict[str, object]] = []
    for _, row in frame.iterrows():
        state: AgentState = make_agent_state(row)
        final_state: AgentState = compiled_graph.invoke(state)
        rows.append(
            {
                "topic": row.get("topic", ""),
                "problem_statement": row.get("problem_statement", ""),
                "predicted_option": final_state.get("answer_option", 5),
                "predicted_solution": " ".join(final_state.get("reasoning", [])).strip(),
                "confidence": final_state.get("confidence", 0.0),
            }
        )
    predictions = pd.DataFrame(rows)
    if output_csv:
        predictions.to_csv(output_csv, index=False)
    return predictions


def main() -> None:
    args = parse_args()
    results = run_pipeline(args.input, args.output, args.model)
    print(results.head())


if __name__ == "__main__":
    main()
