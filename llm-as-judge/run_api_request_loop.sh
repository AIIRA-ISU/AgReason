#!/usr/bin/env bash
set -euo pipefail

# Directories
REQUEST_DIR="/work/mech-ai-scratch/zare/Prompt_creation/llm-judge-stage1/data/jsonl_judge_requests_p2"
RESULT_DIR="/work/mech-ai-scratch/zare/Prompt_creation/llm-judge-stage1/data/jsonl_judge_results_100_p2"

# Ensure output dir exists
mkdir -p "$RESULT_DIR"

# Loop through every JSONL request file
for req in "$REQUEST_DIR"/judge_requests_*.jsonl; do
  base=$(basename "$req" .jsonl)
  out="$RESULT_DIR/${base}_results.jsonl"

  if [[ -f "$out" ]]; then
    echo "Skipping $req → $out (already exists)"
    continue
  fi

  echo "Processing $req → $out"
  python api_request_parallel_processor.py \
    --requests_filepath "$req" \
    --save_filepath     "$out" \
    --request_url       https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute 30 \
    --max_tokens_per_minute   625000 \
    --token_encoding_name     cl100k_base \
    --max_attempts            10 \
    --logging_level           20
done

echo "All done."
