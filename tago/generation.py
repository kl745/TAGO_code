"""Backend-neutral prompt construction and Pass@K generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .schema import Record


class TextGenerator(Protocol):
    def generate(self, prompts: Sequence[str], *, n: int = 1, max_tokens: int = 8192) -> Sequence[Sequence[str]]:
        """Return n outputs for every prompt, preserving input order."""


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(record: Record, template: str) -> str:
    return template.replace("{QUESTION}", record.question).replace("{Question}", record.question)


def generate_pass_k(records: Sequence[Record], generator: TextGenerator, template: str, *, k: int = 1, max_tokens: int = 8192) -> dict[str, list[str]]:
    if k < 1:
        raise ValueError("k must be positive")
    prompts = [build_prompt(record, template) for record in records]
    outputs = generator.generate(prompts, n=k, max_tokens=max_tokens)
    if len(outputs) != len(records) or any(len(row) != k for row in outputs):
        raise ValueError("generator must return exactly k outputs for every prompt")
    return {record.record_id: list(row) for record, row in zip(records, outputs)}

