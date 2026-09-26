"""Anonymous reference implementation for evidence-grounded OR modeling."""

from .metrics import binding_validity_rate, key_evidence_recovery
from .reward import composite_reward

__all__ = ["binding_validity_rate", "key_evidence_recovery", "composite_reward"]

