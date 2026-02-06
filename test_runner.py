#!/usr/bin/env python3
"""Run the project test suite through pytest."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import pytest  # type: ignore
    except Exception:
        print(
            "pytest is required to run tests. Install it with: pip install pytest",
            file=sys.stderr,
        )
        return 1

    args = sys.argv[1:] if len(sys.argv) > 1 else ["-q", "tests"]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
