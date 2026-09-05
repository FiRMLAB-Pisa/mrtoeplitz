#!/usr/bin/env bash
# Build the example notebooks from the scripts beside them.
#
# The .py files are the source: they run as scripts, lint with everything else,
# and review as a diff. The .ipynb files are what a reader opens in Colab, and
# they are generated and executed here so their outputs and the figures under
# examples/figures/ always come from the code as it is now.
#
# With --check, verify the notebooks are current without executing them.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL="${KERNEL:-$(basename "$PWD")-examples}"
check=0
[[ "${1:-}" == "--check" ]] && check=1

# Execute against the interpreter the package is installed into, not whatever
# "python3" the notebook machinery finds first: a notebook that fell back to
# another environment would install this package from PyPI and document that
# one instead of the tree it was built from.
if [[ $check -eq 0 ]]; then
    "$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL" >/dev/null
fi

for script in examples/[0-9]*.py; do
    notebook="${script%.py}.ipynb"
    if [[ $check -eq 1 ]]; then
        "$PYTHON_BIN" -m jupytext --from py:percent --to ipynb "$script" -o - \
            | diff -q - <("$PYTHON_BIN" -m jupytext --to ipynb "$notebook" -o -) >/dev/null \
            || { echo "$notebook is behind $script"; exit 1; }
    else
        echo "$script"
        "$PYTHON_BIN" -m jupytext --from py:percent --to ipynb "$script" -o "$notebook" -q
        "$PYTHON_BIN" -m jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name="$KERNEL" \
            --ExecutePreprocessor.timeout=1800 "$notebook" >/dev/null
    fi
done
