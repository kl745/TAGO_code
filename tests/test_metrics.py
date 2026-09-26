from tago.metrics import evidence_is_authentic, key_evidence_recovery
from tago.parser import parse_output
from tago.reward import outcome_reward
from tago.grpo import easy_fraction, group_advantages, retain_group


def test_authenticity_normalizes_whitespace():
    assert evidence_is_authentic("at  most  24 hours", "The limit is at most 24 hours.")


def test_ker_recall_is_weighted():
    score = key_evidence_recovery(["at least 10 units"], ["at least 10 units", "3 machines"], "at least 10 units and 3 machines")
    assert score["recall"] == 0.5
    assert score["f_beta"] > 0.5


def test_parser_requires_all_steps():
    parsed = parse_output("<model>\nSets: S (evidence: \"set S\")\n</model><code>x</code>")
    assert "missing parameters section" in parsed.format_errors


def test_outcome_grading():
    assert outcome_reward(False, False, False) == 0.0
    assert outcome_reward(True, False, False) == 0.2
    assert outcome_reward(True, True, False) == 0.4
    assert outcome_reward(True, True, True) == 1.0


def test_grpo_schedule():
    assert group_advantages([1.0, 3.0])[0] < 0 < group_advantages([1.0, 3.0])[1]
    assert not retain_group([0.4, 0.42], threshold=0.05)
    assert retain_group([0.0, 1.0], threshold=0.05)
    assert easy_fraction(0.1) == 0.6
    assert easy_fraction(0.5) == 0.2
    assert easy_fraction(0.9) == 0.0
