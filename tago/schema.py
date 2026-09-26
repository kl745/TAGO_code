"""Data structures and validation for StepTrace-style records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

STEPS = ("sets", "parameters", "variables", "objective", "constraints")


@dataclass(frozen=True)
class Binding:
    expression: str
    evidence: str


@dataclass(frozen=True)
class Record:
    record_id: str
    question: str
    key_evidence: tuple[str, ...]
    target: str | None = None
    reference_objective: float | None = None
    difficulty: str | None = None


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def parse_record(raw: Mapping[str, Any], index: int = 0) -> Record:
    """Parse the public JSONL record format without retaining unknown metadata."""
    question = _text(raw.get("question", raw.get("en_question")), "question")
    raw_evidence = raw.get("key_evidence", raw.get("evidence", []))
    if not isinstance(raw_evidence, list) or not all(isinstance(x, str) for x in raw_evidence):
        raise ValueError("key_evidence must be a list of strings")
    record_id = str(raw.get("id", index))
    target = raw.get("target")
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a string when present")
    reference = raw.get("reference_objective", raw.get("answer"))
    if reference is not None:
        try:
            reference = float(reference)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_objective must be numeric") from exc
    difficulty = raw.get("difficulty")
    if difficulty is not None and not isinstance(difficulty, str):
        raise ValueError("difficulty must be a string when present")
    return Record(record_id=record_id, question=question, key_evidence=tuple(raw_evidence), target=target, reference_objective=reference, difficulty=difficulty)
