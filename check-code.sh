#!/usr/bin/env bash
# Convenience script for running Travis-like checks.

set -eu
set -x

pycodestyle tests --max-line-length 120

pylint -j 2 --ignore-patterns='.+\.so' --reports no fc

# Run tests, taking coverage.
# Users can specify extra folders as arguments.
pytest -r sx --cov-report=xml  --cov fc --durations=5 fc tests $@
