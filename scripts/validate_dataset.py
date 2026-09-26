#!/usr/bin/env python3
"""Validate public StepTrace-style JSONL records."""

import argparse
import json
from pathlib import Path

from tago.schema import parse_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    good = 0
    errors = 0
    with args.path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                parse_record(json.loads(line), index)
                good += 1
            except (ValueError, json.JSONDecodeError) as exc:
                errors += 1
                print(f"line {index + 1}: {exc}")
    print(f"valid={good} invalid={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

