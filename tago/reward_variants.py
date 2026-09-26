"""Paper-aligned reward ablations for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardVariant:
    name: str
    use_bvr: bool = True
    use_ker: bool = True
    use_format: bool = True
    graded_outcome: bool = True

    def combine(self, *, bvr: float, ker: float, format_score: float, outcome: float) -> float:
        if not self.graded_outcome:
            outcome = 1.0 if outcome == 1.0 else 0.0
        return (bvr if self.use_bvr else 0.0) + (ker if self.use_ker else 0.0) + (format_score if self.use_format else 0.0) + outcome


VARIANTS = {
    "full": RewardVariant("full"),
    "no_bvr": RewardVariant("no_bvr", use_bvr=False),
    "no_ker": RewardVariant("no_ker", use_ker=False),
    "no_format": RewardVariant("no_format", use_format=False),
    "binary_outcome": RewardVariant("binary_outcome", graded_outcome=False),
}

