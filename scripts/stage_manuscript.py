"""Assemble the manuscript folder so that it compiles on its own.

The figures and tables are generated output and live outside the manuscript
folder, which keeps the record of what is generated separate from the record of
what is written. A reader who receives only the manuscript folder needs them
inside it, so this driver copies the current output into

    manuscript/figures/     the figure PDFs the documents include
    manuscript/tables/      the LaTeX tables the documents input

and writes a build file that compiles the main text and the supplementary
information together, resolving the cross-references that run between them.

Run this after regenerating the figures and tables. The copies are derived
files: the generating code remains the source of record, and the copies are
refreshed rather than edited.
"""

from __future__ import annotations

import os
import shutil

from common import FIG_DIR, ROOT, TAB_DIR

MS_DIR = os.path.join(ROOT, "manuscript")
MS_FIG = MS_DIR  # figures are staged flat next to main.tex (no subfolders)
MS_TAB = os.path.join(MS_DIR, "tables")

# The figures each document includes.
FIGURES = (
    "fig1_mechanism.pdf",
    "fig2_premium.pdf",
    "fig3_landscape.pdf",
    "fig4_portfolio.pdf",
    "fig5_risk_structure.pdf",
)

# The generated tables the supplementary information inputs.
TABLES = (
    "sources.tex",
    "calibration.tex",
    "measures.tex",
    "premium.tex",
    "composition.tex",
    "variance_decomposition.tex",
    "sensitivity.tex",
    "validation.tex",
    "claims.tex",
)

BUILD = r"""#!/bin/sh
# Compile the main text and the supplementary information.
#
# The two documents reference each other, so each is run, then both are run
# again once the other's labels exist. Three passes of the main text settle
# its own references and the bibliography. Only the main text carries a
# bibliography; the supplement cites through it.
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error SI.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error SI.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error SI.tex

echo
echo "built: main.pdf and SI.pdf"
"""


def copy_into(names, source, target, label):
    os.makedirs(target, exist_ok=True)
    missing = []
    for name in names:
        src = os.path.join(source, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        shutil.copy2(src, os.path.join(target, name))
        print("  {}  -> {}".format(label, os.path.relpath(os.path.join(target, name), ROOT)))
    return missing


def main():
    print("staging figures")
    missing_fig = copy_into(FIGURES, FIG_DIR, MS_FIG, "figure")

    print("staging tables")
    missing_tab = copy_into(
        TABLES, os.path.join(TAB_DIR, "tex"), MS_TAB, "table "
    )

    path = os.path.join(MS_DIR, "build.sh")
    with open(path, "w") as fh:
        fh.write(BUILD)
    os.chmod(path, 0o755)
    print("  script -> manuscript/build.sh")

    missing = missing_fig + missing_tab
    if missing:
        print()
        print("missing generated output, so the folder is not yet complete:")
        for name in missing:
            print("  " + name)
        print("run the figure and table drivers, then stage again.")
        return 1

    print()
    print("manuscript folder is self-contained: run manuscript/build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
