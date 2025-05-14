import argparse
import json
import re

def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []

    for item in data:
        question = item.get('question', '')
        response = item.get('response', '')

        if response.startswith("<|user|>"):
            response = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', response, flags=re.DOTALL).strip()

        # Check for unbalanced <think> tags
        has_open_tag = '<think>' in response
        has_close_tag = '</think>' in response

        if has_open_tag != has_close_tag:  # One is present, the other is not
            reasoning = None
            answer = None
        else:
            reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            else:
                reasoning = None
                answer = response.strip()

        new_item = {
            'question': question,
            'reasoning': reasoning,
            'answer': answer
        }
        processed.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Process a JSON file to separate reasoning and answer.")
    parser.add_argument('--input', help='Path to the input JSON file')
    parser.add_argument('--output', help='Path to save the processed JSON file')

    args = parser.parse_args()

    process_json(args.input, args.output)

if __name__ == '__main__':
    main()
