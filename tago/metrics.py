"""Evidence authenticity and evidence-chain metrics."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, Iterable

from .parser import ParsedOutput, iter_bindings


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_is_authentic(evidence: str, question: str) -> bool:
    return normalize_whitespace(evidence) in normalize_whitespace(question)


def format_score(parsed: ParsedOutput, checks: int = 0) -> float:
    """Return the paper's bounded structural score in [0, 0.3]."""
    passed = 0
    total = checks or (2 + 5 + sum(1 for _ in iter_bindings(parsed)))
    passed += parsed.model_text is not None
    passed += parsed.code_text is not None
    passed += sum(bool(parsed.sections.get(step)) for step in ("sets", "parameters", "variables", "objective", "constraints"))
    passed += sum(bool(binding.evidence.strip()) for binding in iter_bindings(parsed))
    return 0.3 * passed / total if total else 0.0


def key_evidence_recovery(predicted: Iterable[str], annotated: Iterable[str], question: str, beta_squared: float = 2.0) -> dict[str, float]:
    """Compute one-to-one KER matching after authenticity filtering.

    Similarity uses SequenceMatcher with the paper's 0.75 threshold. Exact
    substring authenticity is checked against the source question first.
    """
    predicted_all = [normalize_whitespace(x) for x in predicted if x and normalize_whitespace(x)]
    gold = [normalize_whitespace(x) for x in annotated if x and normalize_whitespace(x)]
    eligible = [x for x in predicted_all if x in normalize_whitespace(question)]
    pairs = [(SequenceMatcher(None, p, g).ratio(), i, j) for i, p in enumerate(eligible) for j, g in enumerate(gold)]
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for similarity, i, j in pairs:
        if similarity < 0.75 or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    precision = tp / len(predicted_all) if predicted_all else 0.0
    recall = tp / len(gold) if gold else 0.0
    denom = beta_squared * precision + recall
    f_beta = (1 + beta_squared) * precision * recall / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f_beta": f_beta, "true_positive": float(tp), "predicted": float(len(predicted_all)), "gold": float(len(gold))}


def binding_validity_rate(parsed: ParsedOutput, judge: Callable[[str, str], bool] | None = None) -> float:
    """Score evidence-expression bindings using an injected judge.

    The paper uses an LLM judge for numerical consistency, operator direction,
    and entity correspondence. A judge must be supplied for semantic scoring;
    the default only returns zero rather than inventing semantic judgments.
    """
    bindings = list(iter_bindings(parsed))
    if not bindings or judge is None:
        return 0.0
    return sum(bool(judge(binding.expression, binding.evidence)) for binding in bindings) / len(bindings)

