# TAGO Reference Code

Anonymous reference implementation for evidence-grounded optimization modeling.
The package mirrors the method described in the accompanying paper:

- five ordered modeling steps: Sets, Parameters, Variables, Objective, Constraints;
- verbatim evidence spans attached to each modeling decision;
- deterministic evidence authenticity checks;
- Binding Validity Rate (BVR), Key-Evidence Recovery (KER), format reward, and graded outcome reward;
- a composite reward interface suitable for GRPO rollouts.

This repository is intentionally self-contained and lightweight. It does not contain
author names, affiliations, email addresses, local paths, checkpoint identifiers, or
submission metadata. The LLM judge and solver are adapter interfaces: connect them to
the judge and Pyomo execution service used in an experiment without changing the metric
definitions.

## Layout

```text
tago/                 Core parsing, validation, metrics, rewards, and adapters
configs/              Anonymous SFT/RL/evaluation defaults
scripts/              Command-line dataset and prediction scoring tools
prompts/              Anonymous prompt templates
examples/             Small synthetic example with no paper-private data
tests/                Unit tests for the metric implementations
```

## Quick start

Requires Python 3.10+ and the standard library. Install in editable mode:

```bash
python -m pip install -e .
```

Validate a StepTrace-style JSONL file:

```bash
python scripts/validate_dataset.py examples/sample_dataset.jsonl
```

Score model outputs against an annotated JSONL file:

```bash
python scripts/score_predictions.py \
  --data examples/sample_dataset.jsonl \
  --predictions examples/sample_predictions.jsonl
```

The prediction file must contain one JSON object per line with `id` and `output`.
The annotated data format is documented in `tago/schema.py`.

Generation and evaluation adapters are available in `tago/generation.py`,
`tago/evaluation.py`, and `tago/solver.py`. They support repeated sampling and
Pass@K reporting in the style of a solver-in-the-loop benchmark. The local Python
solver adapter is only for trusted experiments; use a container or remote worker
for untrusted model output.

`configs/default.json` records the paper-aligned SFT/RL/evaluation settings. The
reward ablations in `tago/reward_variants.py` cover removal of BVR, KER, format
reward, and graded outcome reward.

## Output contract

The model output is required to contain exactly one `<model>...</model>` block and
one `<code>...</code>` block. The model block contains five named sections. Each
binding includes `(evidence: "...")`, where the cited text must occur verbatim in
the source problem after whitespace normalization.

## Scope and reproducibility

The code implements the measurable protocol and stable interfaces. Training a model,
calling an LLM judge, and executing generated Pyomo programs require external model,
GPU, and solver services and are deliberately kept behind adapters. This prevents the
reference package from silently inventing experimental results or embedding private
infrastructure details.
