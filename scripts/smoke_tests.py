#!/usr/bin/env python3
"""Run lightweight repository checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: str, expect: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"{script} failed with exit code {result.returncode}")
    if expect not in result.stdout:
        print(result.stdout)
        raise SystemExit(f"{script} output did not include expected text: {expect!r}")
    print(f"ok: {script}")


def main() -> int:
    run_script("content_checks.py", expect="ok: content checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
