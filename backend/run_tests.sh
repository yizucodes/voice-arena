#!/bin/bash
# Test runner script for healer tests
# Usage: ./run_tests.sh [test_type]
# test_type: all, unit, integration, quick

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt
pip install -q pytest pytest-asyncio pytest-cov pytest-mock

# Determine test type
TEST_TYPE=${1:-all}

echo -e "${GREEN}Running tests: ${TEST_TYPE}${NC}"
echo "=========================================="

case $TEST_TYPE in
    unit)
        echo "Running unit tests only..."
        pytest test_healer.py -v -m "unit" --tb=short
        ;;
    integration)
        echo "Running integration tests only..."
        pytest test_healer.py -v -m "integration" --tb=short
        ;;
    quick)
        echo "Running quick tests (excluding slow ones)..."
        pytest test_healer.py -v -m "not slow" --tb=short
        ;;
    coverage)
        echo "Running tests with coverage..."
        pytest test_healer.py --cov=healer --cov-report=html --cov-report=term
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    all|*)
        echo "Running all tests..."
        pytest test_healer.py -v --tb=short
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "All tests passed! ✓${NC}"
    echo "=========================================="
else
    echo -e "${RED}=========================================="
    echo "Some tests failed! ✗${NC}"
    echo "=========================================="
    exit 1
fi
