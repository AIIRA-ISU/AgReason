import argparse
import json

def add_metadata(merged_path, input, output):
    # === Load merged metadata ===
    with open(merged_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    # Create lookup by original_question
    merged_lookup = {item["processed_question"]: item for item in merged_data}

    # === Load new answer+reasoning data ===
    with open(input, "r", encoding="utf-8") as f:
        extra_data = json.load(f)

    # === Match and merge only valid pairs ===
    matched_entries = []
    for extra in extra_data:
        q = extra.get("question")
        if q in merged_lookup:
            combined = merged_lookup[q].copy()

            # Remove the 'question' field before merging
            extra_cleaned = {k: v for k, v in extra.items() if k != "question"}

            # Update metadata with cleaned answer/reasoning fields
            combined.update(extra_cleaned)
            matched_entries.append(combined)

    # === Save final result ===
    with open(output, "w", encoding="utf-8") as f:
        json.dump(matched_entries, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(matched_entries)} matched entries to: {output}")

def main():
    parser = argparse.ArgumentParser(description="Add metadata to question-response pairs.")
    parser.add_argument('--merged_path', default="/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/80k_with_metadata_filtered.json", help='Path to the existing merged JSON file')
    parser.add_argument('--input', help='Path to the new JSON file with question, answer, etc.')
    parser.add_argument('--output', help='Path to save the merged output JSON')

    args = parser.parse_args()

    add_metadata(args.merged_path, args.input, args.output)

if __name__ == '__main__':
    main()
