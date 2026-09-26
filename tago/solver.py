"""Solver execution contracts and a deliberately explicit subprocess adapter."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SolveResult:
    executed: bool
    feasible: bool
    objective: float | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class SolverAdapter(Protocol):
    def solve(self, code: str, *, timeout_seconds: int = 60) -> SolveResult:
        """Execute code in an externally controlled solver environment."""


def run_python_adapter(code: str, *, timeout_seconds: int = 60) -> SolveResult:
    """Minimal local adapter for experiments with trusted generated code.

    This is not a security sandbox. Production evaluation should use a container
    or a remote worker with resource and filesystem restrictions. The generated
    program should print `TAGO_OBJECTIVE=<number>` and `TAGO_FEASIBLE=1`.
    """
    with tempfile.TemporaryDirectory(prefix="tago_eval_") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout_seconds, check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            return SolveResult(False, False, None, str(exc.stdout or ""), str(exc.stderr or ""), True)
    objective = None
    feasible = False
    for line in completed.stdout.splitlines():
        if line.startswith("TAGO_OBJECTIVE="):
            try:
                objective = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        if line.strip() == "TAGO_FEASIBLE=1":
            feasible = True
    return SolveResult(completed.returncode == 0, feasible, objective, completed.stdout, completed.stderr)

