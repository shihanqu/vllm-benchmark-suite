import json
import os

def consolidate_results():
    consolidated_file = "outputs/benchmark_v3_consolidated.json"
    
    with open(consolidated_file, 'r') as f:
        data = json.load(f)
    
    concurrencies = data["concurrencies"]
    contexts = data["contexts"]
    
    files_to_add = {
        "200W": "outputs/benchmark_200W_MiniMax-M2.5-NVFP4_20260215_171355.json",
        "600W": "outputs/benchmark_600W_MiniMax-M2.5-NVFP4_20260215_174212.json"
    }
    
    for wattage, filepath in files_to_add.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            new_data = json.load(f)
            
        # Create a mapping for quick lookup: (ctx, users) -> tokens_per_second
        results_map = {}
        for res in new_data["results"]:
            results_map[(res["context_length"], res["concurrent_users"])] = round(res["tokens_per_second"], 2)
            
        # Build the 2D array [contexts][concurrencies]
        wattage_results = []
        for ctx in contexts:
            ctx_results = []
            for users in concurrencies:
                tps = results_map.get((ctx, users), 0.0)
                ctx_results.append(tps)
            wattage_results.append(ctx_results)
            
        data["wattages"][wattage] = wattage_results
        print(f"Added {wattage} results.")

    # Sort wattages by numeric value (optional but nice)
    sorted_wattages = {}
    for k in sorted(data["wattages"].keys(), key=lambda x: int(x.replace("W", ""))):
        sorted_wattages[k] = data["wattages"][k]
    data["wattages"] = sorted_wattages

    with open(consolidated_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {consolidated_file}")

if __name__ == "__main__":
    consolidate_results()
