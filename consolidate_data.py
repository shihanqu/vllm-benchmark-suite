import json
import os
import re

def consolidate_benchmarks():
    output_dir = "/home/shihan/Projects/vllm-benchmark-suite/vllm-benchmark-suite/outputs"
    files = [f for f in os.listdir(output_dir) if f.endswith('.json') and 'W_MiniMax' in f]
    
    # Sort files by wattage numerically
    def get_wattage(filename):
        match = re.search(r'(\d+)W', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_wattage)
    
    consolidated = {
        "metadata": {
            "model": "MiniMax-M2.5-NVFP4",
            "gpus": "2x NVIDIA RTX PRO 6000 Blackwell (192GB)",
            "output_tokens": 500
        },
        "concurrencies": [],
        "contexts": [],
        "wattages": {}
    }
    
    for filename in files:
        wattage = f"{get_wattage(filename)}W"
        with open(os.path.join(output_dir, filename), 'r') as f:
            data = json.load(f)
            
        if not consolidated["concurrencies"]:
            consolidated["concurrencies"] = data["metadata"]["configuration"]["concurrent_users"]
            consolidated["contexts"] = data["metadata"]["configuration"]["context_lengths"]
            
        # Build Z matrix (rows = contexts, cols = concurrencies)
        # We use a dict lookup for robustness
        lookup = {(r["concurrent_users"], r["context_length"]): r["tokens_per_second"] for r in data["results"]}
        
        z_matrix = []
        for ctx in consolidated["contexts"]:
            row = []
            for conc in consolidated["concurrencies"]:
                row.append(round(lookup.get((conc, ctx), 0), 2))
            z_matrix.append(row)
            
        consolidated["wattages"][wattage] = z_matrix

    output_path = os.path.join(output_dir, "benchmark_v3_consolidated.json")
    with open(output_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"Consolidated data saved to: {output_path}")

if __name__ == "__main__":
    consolidate_benchmarks()
