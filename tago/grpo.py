"""Small, framework-independent pieces of the paper's GRPO schedule."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Normalize rewards within one prompt group as in GRPO."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    scale = math.sqrt(variance) + epsilon
    return [(value - mean) / scale for value in rewards]


def retain_group(rewards: Sequence[float], threshold: float = 0.05) -> bool:
    """Keep a prompt only when its reward standard deviation is informative."""
    if not rewards:
        return False
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return std > threshold


def easy_fraction(progress: float) -> float:
    """Piecewise easy-prompt fraction: 60% -> 20% -> 0% across training."""
    progress = min(1.0, max(0.0, progress))
    if progress < 0.30:
        return 0.60
    if progress < 0.70:
        return 0.20
    return 0.0

