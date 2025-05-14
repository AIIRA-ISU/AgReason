
# Creating the responses to all the generated 54K questions from the previous module
python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/new_filtered_generation/flowchart/kept_questions.xlsx" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/output/response_llama4.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/output/partial_path_llama4.json" --model "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_deepseek_v3_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_deepseek_v3_gbm.json" --model "deepseek-ai/DeepSeek-V3" --from_benchmark

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_claude_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_claude_gbm.jsonl" --model "claude-3-7-sonnet-20250219" --provider "claude" --from_benchmark

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gpto1_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_gpto1_gbm.jsonl" --model "o1-2024-12-17" --provider "gpt" --from_benchmark

python main.ipynb --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gemini_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_gemini_gbm.jsonl" --model "gemini-2.5-flash-preview-04-17" --provider "gemini" --from_benchmark

python benchmark_stats_sampling.ipynb --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_phi3_gbm.jsonl" --model microsoft/phi-3-mini-4k-instruct --from_benchmark

python "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/raw_hf.py" --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_sft_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_phi3_sft_gbm.jsonl" --model "/work/mech-ai-scratch/shreyang/Agentic Ag/distillation/llama-3.1-8b-deepseek-finetuned-final" --from_benchmark

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gpt4o_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_gpt4o_gbm.jsonl" --model "gpt-4o-2024-08-06" --provider "gpt" --from_benchmark

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_claude_3_opus_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_claude_3_opus_gbm.jsonl" --model "claude-3-opus-20240229" --provider "claude" --from_benchmark --not_thinking

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama4_scout_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_llama4_scout_gbm.json" --model "meta-llama/Llama-4-Scout-17B-16E-Instruct" --from_benchmark

python main.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_24b_gbm.json" --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_mistral_24b_gbm.json" --model "mistralai/Mistral-Small-24B-Instruct-2501" --from_benchmark


# Formatting the responses into reasoning traces and answers
python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/response_llama4.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama4.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_deepseek_v3_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_deepseek_v3_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_claude_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_claude_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gpto1_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gpto1_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gemini_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gemini_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_sft_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_sft_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_gpt4o_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gpt4o_gbm.json"         

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_claude_3_opus_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_claude_3_opus_gbm.json"    

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama4_scout_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama4_scout_gbm.json"

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_24b_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_24b_gbm.json" 

# Merging the formatted reasoning traces and answers and questions with metadata
python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama4.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama4.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_deepseek_v3_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_deepseek_v3_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_claude_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_claude_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gpto1_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gpto1_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gemini_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gemini_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_sft_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_sft_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_gpt4o_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gpt4o_gbm.json"  

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_claude_3_opus_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_claude_3_opus_gbm.json"  

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama4_scout_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama4_scout_gbm.json"

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_24b_gbm.json" --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_24b_gbm.json" 

# Selecting the smaller benchmark set out of the previously merged questions
python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama4.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/llama4.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_deepseek_v3_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/deepseek_v3_gbm.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_claude_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/claude.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gpto1_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/gpto1.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gemini_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/gemini.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/phi3_gbm.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_sft_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/phi3_sft_gbm.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_gpt4o_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/gpt4o_gbm.json"  

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_claude_3_opus_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/claude_3_opus_gbm.json"  

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama4_scout_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/llama4_scout_gbm.json"

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions.json" --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_24b_gbm.json" --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/mistral_24b_gbm.json" 