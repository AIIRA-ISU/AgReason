python main.py \
  --input_file "/u/sganguly1/AgReason/distillation/final_annotation_questions.json" \
  --output_file results.json \
  --partial_file_path partial.jsonl \
  --model microsoft/phi-3-mini-4k-instruct \
  --num_samples 10 \
  --from_benchmark

python distil.py   --data_path "/work/mech-ai-scratch/shreyang/Agentic Ag/llm_judge/output/R1-response/llm_judge_big_ds_results_filtered.json"   --benchmark_path "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json"   --model_id microsoft/phi-3-mini-4k-instruct   --output_dir ./sft_models/phi-3-deepseek-finetuned   --num_train_epochs 5   --eval_steps 1200

python distil.py   --data_path "/work/mech-ai-scratch/shreyang/Agentic Ag/llm_judge/output/R1-response/llm_judge_big_ds_results_filtered.json"   --benchmark_path "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json"   --model_id meta-llama/Llama-3.1-8B   --output_dir ./sft_models/llama-3.1-8b-finetuned   --num_train_epochs 5   --eval_steps 10

python distil.py   --data_path "/work/mech-ai-scratch/shreyang/Agentic Ag/llm_judge/output/R1-response/llm_judge_big_ds_results_filtered.json"   --benchmark_path "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json"   --model_id Qwen/Qwen3-8B   --output_dir ./sft_models/qwen3-8b  --num_train_epochs 5   --eval_steps 5

python distil.py   --data_path "/work/mech-ai-scratch/shreyang/Agentic Ag/llm_judge/output/R1-response/llm_judge_big_ds_results_filtered.json"   --benchmark_path "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json"   --model_id mistralai/Mistral-7B-Instruct-v0.3   --output_dir ./sft_models/mistral-7b-finetuned   --num_train_epochs 3   --eval_steps 10