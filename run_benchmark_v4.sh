#!/usr/bin/env bash
# Launch the v4 benchmark (LLM-Inference-Bench methodology, v3-style reports).
#   ./run_benchmark_v4.sh                              # uses configs/v4/example.toml
#   ./run_benchmark_v4.sh configs/v4/argonne.toml      # any config file
#   ./run_benchmark_v4.sh configs/v4/example.toml --preset quick --label test   # extra CLI flags

# Exit immediately if a command exits with a non-zero status
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== vLLM Benchmark Suite v4 Launch Script ===${NC}"

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
        uv venv venv
        source venv/bin/activate
        uv pip install -r requirements.txt
    else
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

# Step 2: Verify v4 requirements
echo -e "${GREEN}Verifying dependencies are installed...${NC}"
python3 -c "import requests, aiohttp, rich, numpy, pandas, matplotlib, seaborn" || {
    echo -e "${YELLOW}Some dependencies are missing. Installing...${NC}"
    pip install -r requirements.txt
}
python3 -c "import pynvml" 2>/dev/null || echo -e "${YELLOW}nvidia-ml-py not installed: falling back to nvidia-smi (no hardware energy counter).${NC}"

# Step 3: Run
CONFIG="${1:-configs/v4/example.toml}"
shift || true
echo -e "${GREEN}Launching v4 benchmark with ${CONFIG}...${NC}"
python3 vllm_benchmark_suitev4.py run --config "$CONFIG" -y "$@"

echo -e "${GREEN}Benchmark execution completed successfully!${NC}"
