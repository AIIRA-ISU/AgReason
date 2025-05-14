#!/bin/bash
python api_request_parallel_grok.py \
  --requests_filepath /work/mech-ai-scratch/zare/Prompt_creation/answer-creation-grok/data/grok-3-beta/grok-3-beta_processed_questions.jsonl \
  --save_filepath /work/mech-ai-scratch/zare/Prompt_creation/answer-creation-grok/data/grok-3-beta/grok-3-beta_processed_results.jsonl \
  --request_url https://api.x.ai/v1/chat/completions \
  --max_requests_per_minute 180 \
  --max_tokens_per_minute 625000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
