#!/usr/bin/env bash
# Convenience script for running Github workflow checks.

set -euo pipefail
set -x

pycodestyle tests --max-line-length 120

pylint -j 2 --ignore-patterns='.+\.so' --reports no fc

# Run tests, taking coverage.
# Test from temp dir to prevent using the checkout instead of installed code
original_dir="$PWD"
cd /tmp
pytest \
    -r sx \
    --cov=fc \
    --cov-report=term-missing \
    --cov-report="xml:${GITHUB_WORKSPACE}/coverage.xml" \
    --durations=5 \
    "${GITHUB_WORKSPACE}/tests"
cd "$original_dir"
