"""Composite reward matching the paper's four reward components."""

from __future__ import annotations

from typing import Iterable, Callable

from .metrics import binding_validity_rate, format_score, key_evidence_recovery
from .parser import iter_bindings, parse_output


def outcome_reward(executed: bool, feasible: bool, objective_matches: bool) -> float:
    if not executed:
        return 0.0
    if not feasible:
        return 0.2
    return 1.0 if objective_matches else 0.4


def composite_reward(output: str, question: str, key_evidence: Iterable[str], *, executed: bool = False, feasible: bool = False, objective_matches: bool = False, binding_judge: Callable[[str, str], bool] | None = None) -> dict[str, float]:
    parsed = parse_output(output)
    bindings = [binding.evidence for binding in iter_bindings(parsed)]
    ker = key_evidence_recovery(bindings, key_evidence, question)
    components = {
        "bvr": binding_validity_rate(parsed, binding_judge),
        "ker": ker["f_beta"],
        "format": format_score(parsed),
        "outcome": outcome_reward(executed, feasible, objective_matches),
    }
    components["total"] = sum(components.values())
    return components


