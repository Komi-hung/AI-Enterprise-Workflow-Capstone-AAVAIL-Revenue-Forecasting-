"""
run_tests.py
============
Standalone test runner for the AAVAIL Capstone Project.

Usage:
    python run_tests.py                   # run all tests
    python run_tests.py --module api      # run only test_api.py
    python run_tests.py --module model    # run only test_model.py
    python run_tests.py --module logging  # run only test_logging.py
    python run_tests.py --coverage        # run with HTML coverage report

This script is intentionally framework-agnostic — it simply delegates to
pytest so that it can be invoked without a pytest config file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Map short module aliases → test file paths
MODULE_MAP: dict[str, str] = {
    "api":     "tests/test_api.py",
    "model":   "tests/test_model.py",
    "logging": "tests/test_logging.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AAVAIL Unit Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py
  python run_tests.py --module api
  python run_tests.py --coverage
  python run_tests.py --module model --verbose
        """,
    )
    parser.add_argument(
        "--module", "-m",
        choices=list(MODULE_MAP.keys()),
        default=None,
        help="Run only the specified test module (default: all).",
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate an HTML coverage report in htmlcov/.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Pass -v flag to pytest for verbose output.",
    )
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure.",
    )
    return parser.parse_args()


def build_pytest_command(args: argparse.Namespace) -> list[str]:
    """Construct the full pytest command list from parsed arguments."""
    cmd = [sys.executable, "-m", "pytest"]

    # Test target(s)
    if args.module:
        cmd.append(str(_ROOT / MODULE_MAP[args.module]))
    else:
        cmd.append(str(_ROOT / "tests"))

    # Optional flags
    if args.verbose:
        cmd.append("-v")

    if args.failfast:
        cmd.append("-x")

    if args.coverage:
        cmd += [
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ]

    # Always show short test summary
    cmd += ["-r", "a"]

    return cmd


def main() -> int:
    args = _parse_args()
    cmd  = build_pytest_command(args)

    print("=" * 60)
    print("AAVAIL Revenue Forecasting — Test Runner")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_ROOT))

    print()
    if result.returncode == 0:
        print("[PASS] All tests passed.")
    else:
        print(f"[FAIL] Tests failed (exit code {result.returncode}).")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
