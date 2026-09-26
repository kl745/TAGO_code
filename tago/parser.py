"""Parser for the model-then-code output contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import Binding, STEPS

_BLOCK = re.compile(r"<(model|code)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_EVIDENCE = re.compile(r"\(\s*evidence\s*:\s*[\"“](.*?)[\"”]\s*\)", re.IGNORECASE)
_HEADING = re.compile(r"^\s*(sets|parameters|variables|objective|constraints)\s*:\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedOutput:
    model_text: str | None
    code_text: str | None
    sections: dict[str, tuple[Binding, ...]]
    format_errors: tuple[str, ...]


def _parse_sections(text: str) -> tuple[dict[str, tuple[Binding, ...]], list[str]]:
    sections: dict[str, list[Binding]] = {name: [] for name in STEPS}
    errors: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1).lower()
            expression = heading.group(2).strip()
            if expression:
                evidence = _EVIDENCE.search(expression)
                if evidence:
                    sections[current].append(Binding(expression=expression[: evidence.start()].strip(), evidence=evidence.group(1)))
                else:
                    errors.append(f"missing evidence in {current}")
            continue
        if current is None or not line.strip():
            continue
        evidence = _EVIDENCE.search(line)
        if evidence:
            expression = line[: evidence.start()].strip(" -\t")
            if expression:
                sections[current].append(Binding(expression=expression, evidence=evidence.group(1)))
        elif line.strip():
            errors.append(f"missing evidence in {current}")
    return {key: tuple(value) for key, value in sections.items()}, errors


def parse_output(output: str | None) -> ParsedOutput:
    if not output:
        return ParsedOutput(None, None, {key: tuple() for key in STEPS}, ("empty output",))
    blocks = {match.group(1).lower(): match.group(2).strip() for match in _BLOCK.finditer(output)}
    errors: list[str] = []
    if "model" not in blocks:
        errors.append("missing <model> block")
    if "code" not in blocks:
        errors.append("missing <code> block")
    sections, section_errors = _parse_sections(blocks.get("model", ""))
    errors.extend(section_errors)
    for step in STEPS:
        if not sections[step]:
            errors.append(f"missing {step} section")
    return ParsedOutput(blocks.get("model"), blocks.get("code"), sections, tuple(dict.fromkeys(errors)))


def iter_bindings(parsed: ParsedOutput) -> Iterable[Binding]:
    for step in STEPS:
        yield from parsed.sections.get(step, ())

