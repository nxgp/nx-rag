#!/usr/bin/env bash
# Developer Verification Script: Runs linter, formatter, and unit tests.

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Use active virtual environment binaries if available
RUFF="ruff"
PYTEST="pytest"

if ! command -v ruff &> /dev/null && [ -f ".venv/bin/ruff" ]; then
    RUFF=".venv/bin/ruff"
fi

if ! command -v pytest &> /dev/null && [ -f ".venv/bin/pytest" ]; then
    PYTEST=".venv/bin/pytest"
fi

echo -e "${GREEN}=== 1/3 Running Ruff Linter ===${NC}"
$RUFF check src tests scripts

echo -e "\n${GREEN}=== 2/3 Running Ruff Formatter (Auto-Fixing) ===${NC}"
$RUFF format src tests scripts

echo -e "\n${GREEN}=== 3/3 Running Pytest Unit Tests ===${NC}"
$PYTEST -v

echo -e "\n${GREEN}===========================================${NC}"
echo -e "${GREEN}  ALL CHECKS PASSED SUCCESSFULLY!  ${NC}"
echo -e "${GREEN}===========================================${NC}"
