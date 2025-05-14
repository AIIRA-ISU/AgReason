import json

# === FILE PATHS ===
json_metadata_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_generations/flowchart/80k_with_metadata.json"
jsonl_processed_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/partial_results.jsonl"
output_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/80k_with_metadata.json"

# === STEP 1: Load metadata JSON ===
with open(json_metadata_path, "r", encoding="utf-8") as f:
    metadata_data = json.load(f)

# Build a mapping from original_question -> metadata object
metadata_map = {entry['original_question']: entry for entry in metadata_data}

# === STEP 2: Read JSONL and update matching entries ===
with open(jsonl_processed_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            json_obj = json.loads(line)
            orig_q = json_obj.get("original_question")
            proc_q = json_obj.get("processed_question")

            if orig_q in metadata_map:
                metadata_map[orig_q]["processed_question"] = proc_q
        except json.JSONDecodeError as e:
            print("Skipping malformed line:", e)

# === STEP 3: Write merged output ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(metadata_map.values()), f, indent=2, ensure_ascii=False)

print(f"✅ Merged output saved to: {output_path}")