import json
import pandas as pd

# === Input & Output Paths ===
input_json_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/80k_with_metadata_merged.json"
output_json_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/80k_with_metadata_filtered.json"
output_excel_path = "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/80k_with_metadata_filtered.xlsx"

# === Load JSON ===
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Filter Entries ===
filtered = []
for item in data:
    proc_q = item.get("processed_question")
    tpl_col = item.get("template_column")

    # Condition 1: processed_question must not be null
    if proc_q is None:
        continue

    # Condition 2: processed_question must not start with {'decision': 'False',
    is_bad_proc_q = isinstance(proc_q, dict) and proc_q.get("decision") == "False"

    # Condition 3: template_column must not be blank or missing
    is_blank_template = not tpl_col or str(tpl_col).strip() == ""

    # Keep only if all conditions are met
    if not is_bad_proc_q and not is_blank_template:
        filtered.append(item)

# === Save Filtered JSON ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"✅ Filtered JSON saved to: {output_json_path}")

# === Save as Excel ===
df = pd.DataFrame(filtered)
df.to_excel(output_excel_path, index=False)
print(f"✅ Filtered Excel saved to: {output_excel_path}")
