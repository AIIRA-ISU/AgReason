import argparse
import json

def merge_answers(benchmark_path, larger_path, output_path):
    # Load benchmark (subset)
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # Load larger benchmark (full set)
    with open(larger_path, 'r', encoding='utf-8') as f:
        larger_data = json.load(f)

    # Create a lookup from the larger set based on the question
    larger_lookup = {item['processed_question']: item for item in larger_data}

    matched_entries = []
    match_count = 0

    for benchmark_item in benchmark_data:
        question = benchmark_item.get('processed_question')
        if question in larger_lookup:
            matched_large_item = larger_lookup[question]
            # Copy "reasoning" and "answer" fields
            reasoning = matched_large_item.get('reasoning')
            answer = matched_large_item.get('answer')
            # Add to the benchmark question
            benchmark_item['reasoning'] = reasoning
            benchmark_item['answer'] = answer
            match_count += 1
        matched_entries.append(benchmark_item)

    # Save the new benchmark dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matched_entries, f, indent=2, ensure_ascii=False)

    print(f"✅ {match_count} questions matched and updated.")
    print(f"📄 Saved the updated benchmark to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Attach reasoning and answer to a benchmark JSON from a larger dataset.")
    parser.add_argument('--benchmark_json', help='Path to the benchmark JSON file (subset questions)')
    parser.add_argument('--larger_json', help='Path to the larger JSON file (full benchmark with answers)')
    parser.add_argument('--output_json', help='Path to save the updated benchmark JSON file')

    args = parser.parse_args()

    merge_answers(args.benchmark_json, args.larger_json, args.output_json)

if __name__ == '__main__':
    main()
