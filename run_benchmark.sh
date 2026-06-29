#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== vLLM Benchmark Suite Launch Script ===${NC}"

# Navigate to the repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Detect/Create virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}Detecting existing virtual environment 'venv'...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}Detecting existing virtual environment '.venv'...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}Creating venv using 'uv'...${NC}"
        uv venv venv
        source venv/bin/activate
        echo -e "${GREEN}Installing dependencies using 'uv'...${NC}"
        uv pip install -r requirements.txt
    else
        echo -e "${GREEN}Creating venv using standard 'venv' module...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        echo -e "${GREEN}Installing dependencies using 'pip'...${NC}"
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

# Step 2: Verify requirements are satisfied
echo -e "${GREEN}Verifying dependencies are installed...${NC}"
python3 -c "import requests, rich, numpy, pandas, matplotlib, seaborn" || {
    echo -e "${YELLOW}Some dependencies are missing. Installing...${NC}"
    pip install -r requirements.txt
}

# Step 3: Run the benchmark
echo -e "${GREEN}Launching non-interactive v3 benchmark...${NC}"
python3 run_benchmark_v3.py

echo -e "${GREEN}Benchmark execution completed successfully!${NC}"
