from __future__ import annotations

import textwrap
from typing import Callable, Dict

from .prompt import PromptRunner

LIST_TEMPLATE = '[2, ["Reason step 1", "Reason step 2"], 0.72]'
GUIDANCE_TEXT = (
    "Return exactly one JSON list: [option_number, [reason_steps...], confidence_between_0_and_1]."
)


def make_sequence_solver(run_prompt: PromptRunner) -> Callable[[Dict[str, object]], Dict[str, object]]:
    def sequence_solver(state: Dict[str, object]) -> Dict[str, object]:
        query = str(state.get("query", ""))
        options = state.get("options", []) or []
        if isinstance(options, list):
            options_text = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options[:5], 1)) or "No options provided"
        else:
            options_text = "No options provided"

        prompt = textwrap.dedent(
            f"""
            You are a mathematical sequence analyst. Determine the next sensible element and map it to the provided choices.

            Sequence:
            {query}

            Options:
            {options_text}

            Respond with a single JSON list following this example:
            {LIST_TEMPLATE}
            """
        ).strip()

        run_prompt(state, prompt, guidance=GUIDANCE_TEXT)
        return state

    return sequence_solver


def make_classic_riddle(run_prompt: PromptRunner) -> Callable[[Dict[str, object]], Dict[str, object]]:
    def classic_riddle(state: Dict[str, object]) -> Dict[str, object]:
        query = str(state.get("query", ""))
        options = state.get("options", []) or []
        if isinstance(options, list):
            options_text = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options[:5], 1)) or "No options provided"
        else:
            options_text = "No options provided"

        prompt = textwrap.dedent(
            f"""
            You are a classic riddle expert. Explain the riddle and pick the best option.

            Riddle:
            {query}

            Options:
            {options_text}

            Respond with a single JSON list following this example:
            {LIST_TEMPLATE}
            """
        ).strip()

        run_prompt(state, prompt, guidance=GUIDANCE_TEXT)
        return state

    return classic_riddle


def make_other_task(run_prompt: PromptRunner) -> Callable[[Dict[str, object]], Dict[str, object]]:
    def other_task(state: Dict[str, object]) -> Dict[str, object]:
        query = str(state.get("query", ""))
        options = state.get("options", []) or []
        if isinstance(options, list):
            options_text = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options[:5], 1)) or "No options provided"
        else:
            options_text = "No options provided"

        prompt = textwrap.dedent(
            f"""
            You are a general reasoning assistant. Inspect each option and choose the best.

            Question:
            {query}

            Options:
            {options_text}

            Respond with a single JSON list following this example:
            {LIST_TEMPLATE}
            """
        ).strip()

        run_prompt(state, prompt, guidance=GUIDANCE_TEXT)
        return state

    return other_task
