from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import OllamaLLM


class PromptRunner:
    """Call the model once and extract [option, reasoning, confidence]."""

    def __init__(self, model: str) -> None:
        # Initialise the Ollama client and stash the last raw reply for debugging.
        self.llm = OllamaLLM(model=model)
        self.last_raw: Optional[str] = None

    def __call__(
        self,
        state: dict[str, Any],
        prompt: str,
        guidance: Optional[str] = None,
    ) -> Tuple[int, List[str], float]:
        # Build the full prompt, invoke LLM, parse the triplet, and update the agent state.
        message = prompt if not guidance else f"{prompt}\n\nAdditional guidance:\n{guidance}"
        raw_text = self._coerce(self.llm.invoke(message))
        option, reasoning, confidence = self._extract(raw_text)
        state["answer_option"], state["reasoning"], state["confidence"] = option, reasoning, confidence
        self.last_raw = raw_text
        return option, reasoning, confidence

    def _extract(self, raw_text: str) -> Tuple[int, List[str], float]:
        # Slice out the bracketed payload, interpret it, and return clamped values + reasoning list.
        snippet = raw_text.strip()
        candidate = self._slice_payload(snippet)
        triplet = self._parse_triplet(candidate)
        if triplet is None:
            fallback = snippet[:200] or "Unable to parse model response."
            return 5, [fallback], 0.0
        option_val, reasoning_val, confidence_val = triplet
        option = self._clamp_int(option_val, 1, 5, default=5)
        confidence = self._clamp_float(confidence_val, 0.0, 1.0, default=0.0)
        reasoning = self._clean_reasoning(reasoning_val)
        return option, reasoning, confidence

    def _parse_triplet(self, candidate: str) -> Optional[Tuple[Any, Any, Any]]:
        # Try JSON first, then literal_eval, accepting list/tuple triples or dict payloads.
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(candidate)
            except Exception:
                continue
            if isinstance(data, (list, tuple)) and len(data) == 3:
                return tuple(data)
            if isinstance(data, dict):
                return self._extract_from_dict(data)
        return None

    @staticmethod
    def _extract_from_dict(data: Dict[str, Any]) -> Optional[Tuple[Any, Any, Any]]:
        # Map common dict keys into the expected triple layout.
        key_map = (
            data.get("answer_option"),
            data.get("option"),
            data.get("predicted_option"),
        )
        option_val = next((value for value in key_map if value is not None), None)
        if option_val is None:
            return None
        reasoning_val = data.get("reasoning") or data.get("reasons") or data.get("explanation")
        confidence_val = data.get("confidence") or data.get("score") or data.get("probability")
        return option_val, reasoning_val, confidence_val

    @staticmethod
    def _clean_reasoning(value: Any) -> List[str]:
        # Normalise reasoning into a stripped list of lines or bullets.
        if isinstance(value, (list, tuple)):
            steps = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            lines = text.splitlines() if text else []
            steps = [line.strip(" -*\t") for line in lines if line.strip(" -*\t")]
        return steps or ["Reasoning not provided."]

    @staticmethod
    def _clamp_int(value: Any, low: int, high: int, *, default: int) -> int:
        # Coerce to int and clamp to inclusive bounds, falling back to a default.
        try:
            number = int(str(value).strip())
        except Exception:
            number = default
        return max(low, min(high, number))

    @staticmethod
    def _clamp_float(value: Any, low: float, high: float, *, default: float) -> float:
        # Coerce to float and clamp to inclusive bounds, falling back to a default.
        try:
            number = float(value)
        except Exception:
            number = default
        return max(low, min(high, number))

    @staticmethod
    def _coerce(raw: Any) -> str:
        # Flatten LangChain/Ollama responses into plain text for parsing.
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "generations"):
            try:
                return raw.generations[0][0].text
            except Exception:
                pass
        content = getattr(raw, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            try:
                return "".join(part.get("text", "") for part in content)
            except Exception:
                pass
        return str(raw)

    @staticmethod
    def _slice_payload(snippet: str) -> str:
        # Capture the first balanced bracketed block; fall back to the original snippet.
        start = snippet.find("[")
        if start == -1:
            return snippet
        depth = 0
        for idx in range(start, len(snippet)):
            char = snippet[idx]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return snippet[start : idx + 1]
        return snippet[start:]
