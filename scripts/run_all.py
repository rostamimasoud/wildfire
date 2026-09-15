"""Reproduce every figure, table and validation result, in order.

Run from the ``scripts`` directory with ``python3 run_all.py``. Each stage is
independent of the others except through the cached scenario solutions, so a
single stage can be re-run on its own after an edit.
"""

from __future__ import annotations

import subprocess
import sys
import time

STAGES = [
    ("validation", "validation.py"),
    ("tables", "make_tables.py"),
    ("figure 1, the mechanism", "fig1_mechanism.py"),
    ("figure 2, the premium and the ceiling", "fig2_premium.py"),
    ("figure 3, the portfolio and the schedule", "fig3_portfolio.py"),
]


def main() -> int:
    failures = []
    for label, script in STAGES:
        print("=" * 74)
        print(label)
        print("=" * 74)
        start = time.time()
        result = subprocess.run([sys.executable, script])
        elapsed = time.time() - start
        if result.returncode != 0:
            failures.append(label)
            print("  FAILED after {:.0f} s".format(elapsed))
        else:
            print("  done in {:.0f} s".format(elapsed))
    print("=" * 74)
    if failures:
        print("failed stages: " + ", ".join(failures))
        return 1
    print("all stages completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
