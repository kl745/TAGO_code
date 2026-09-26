"""Pass@K evaluation with solver and evidence-chain metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .metrics import binding_validity_rate, key_evidence_recovery
from .parser import iter_bindings, parse_output
from .schema import Record
from .solver import SolveResult, SolverAdapter


@dataclass(frozen=True)
class CandidateResult:
    correct: bool
    output: str
    solver: SolveResult


@dataclass(frozen=True)
class RecordResult:
    record_id: str
    pass_k: bool
    candidates: tuple[CandidateResult, ...]


def _matches(value: float | None, target: float | None, tolerance: float) -> bool:
    if value is None or target is None:
        return False
    return abs(value - target) <= tolerance * max(1.0, abs(target))


def evaluate_pass_k(records: Sequence[Record], outputs: Mapping[str, Sequence[str]], solver: SolverAdapter, *, tolerance: float = 1e-4, timeout_seconds: int = 60) -> tuple[list[RecordResult], float]:
    results: list[RecordResult] = []
    for record in records:
        candidates: list[CandidateResult] = []
        for output in outputs.get(record.record_id, ()):
            parsed = parse_output(output)
            solve = solver.solve(parsed.code_text or "", timeout_seconds=timeout_seconds) if parsed.code_text else SolveResult(False, False, None)
            candidates.append(CandidateResult(solve.executed and solve.feasible and _matches(solve.objective, record.reference_objective, tolerance), output, solve))
        results.append(RecordResult(record.record_id, any(item.correct for item in candidates), tuple(candidates)))
    accuracy = sum(item.pass_k for item in results) / len(results) if results else 0.0
    return results, accuracy

