#!/usr/bin/env bash
# Convenience script for running Travis-like checks.

set -eu
set -x

pep8 tests --max-line-length 120

pylint -j 2 --ignore-patterns='.+\.so' --reports no fc

# Run tests, taking coverage.
# Users can specify extra folders as arguments.
py.test -r sx --cov fc --durations=5 fc tests $@

