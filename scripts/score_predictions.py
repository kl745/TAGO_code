#!/usr/bin/env python3
"""Score one generated output per annotated JSONL record."""

import argparse
import json
from pathlib import Path

from tago.reward import composite_reward
from tago.schema import parse_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prediction_lines = (line for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip())
    predictions = {str(row["id"]): row["output"] for row in map(json.loads, prediction_lines) if row.get("output") is not None}
    rows = []
    with args.data.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = parse_record(json.loads(line), index)
            scores = composite_reward(predictions.get(record.record_id, ""), record.question, record.key_evidence)
            rows.append({"id": record.record_id, **scores})
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else "")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
