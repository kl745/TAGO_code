"""Five-stage candidate validation with external services behind protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .metrics import evidence_is_authentic
from .parser import ParsedOutput, iter_bindings, parse_output
from .schema import STEPS


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    stages: dict[str, bool]
    errors: tuple[str, ...]


def validate_candidate(
    question: str,
    output: str,
    *,
    solver_check: Callable[[str], bool] | None = None,
    alignment_check: Callable[[str, str], bool] | None = None,
    model_code_check: Callable[[ParsedOutput], bool] | None = None,
) -> ValidationResult:
    """Apply the paper's five filters in order.

    Stage 4 and 5 are intentionally callbacks: an LLM judge and a Pyomo
    runtime are experiment-dependent and should not be silently replaced by
    a heuristic.
    """
    parsed = parse_output(output)
    stages = {"format": not parsed.format_errors, "authenticity": False, "solver": False, "evidence_math": False, "model_code": False}
    errors = list(parsed.format_errors)
    if not stages["format"]:
        return ValidationResult(False, stages, tuple(errors))

    bindings = list(iter_bindings(parsed))
    stages["authenticity"] = all(evidence_is_authentic(binding.evidence, question) for binding in bindings)
    if not stages["authenticity"]:
        errors.append("one or more evidence spans are not authentic substrings")
        return ValidationResult(False, stages, tuple(errors))

    stages["solver"] = bool(solver_check and parsed.code_text and solver_check(parsed.code_text))
    if not stages["solver"]:
        errors.append("solver validation did not pass")
        return ValidationResult(False, stages, tuple(errors))

    stages["evidence_math"] = bool(alignment_check and all(alignment_check(item.expression, item.evidence) for item in bindings))
    if not stages["evidence_math"]:
        errors.append("evidence-math alignment did not pass")
        return ValidationResult(False, stages, tuple(errors))

    stages["model_code"] = bool(model_code_check and model_code_check(parsed))
    if not stages["model_code"]:
        errors.append("model-code alignment did not pass")
    return ValidationResult(all(stages.values()), stages, tuple(errors))

