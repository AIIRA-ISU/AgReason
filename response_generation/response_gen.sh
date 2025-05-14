#!/bin/bash

# Copy/paste this job script into a text file and submit with:
#    sbatch llama3distil.sh

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --job-name="response_gen"
#SBATCH --mail-user=shreyang@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="sbatch-logs/response"
#SBATCH --error="sbatch-logs/response-err"

# Activate conda (adjust path if needed)
#/work/mech-ai/shreyan/minconda3/condabin/conda init
conda activate /work/mech-ai/shreyan/miniconda3/envs/ag_prompt

# Change to working dir (quote paths!)
cd "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation"

huggingface-cli login --token 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

# Run your Python script with arguments

#LLama 3 8B SFT GBM
python raw_hf.py  --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama3_8b_sft_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_llama3_8b_sft_gbm.json"\
                --model "/work/mech-ai-scratch/shreyang/Agentic Ag/distillation/sft_models/llama-3.1-8b-finetuned/checkpoint-12000/merged" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama3_8b_sft_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama3_8b_sft_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama3_8b_sft_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama3_8b_sft_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama3_8b_sft_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/llama3_8b_sft_gbm.json" 

# LLama 3 8B without SFT GBM
python raw_hf.py  --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama3_8b_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_llama3_8b_gbm.json"\
                --model "meta-llama/Llama-3.1-8B" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_llama3_8b_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama3_8b_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_llama3_8b_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama3_8b_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_llama3_8b_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/llama3_8b_gbm.json" 


#Mistral-7b SFT GBM
python raw_hf.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_7b_sft_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_mistral_7b_sft_gbm.json"\
                --model "/work/mech-ai-scratch/shreyang/Agentic Ag/distillation/sft_models/mistral-7b-finetuned/checkpoint-12000/merged" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_7b_sft_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_7b_sft_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_7b_sft_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_7b_sft_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_7b_sft_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/mistral_7b_sft_gbm.json" 

# Mistral-7B without SFT GBM
python raw_hf.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_7b_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_mistral_7b_gbm.json"\
                --model "mistralai/Mistral-7B-Instruct-v0.3" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_mistral_7b_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_7b_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_mistral_7b_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_7b_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_mistral_7b_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/mistral_7b_gbm.json" 

#Phi3 SFT GBM
python raw_hf.py  --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_sft_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_phi3_sft_gbm.json"\
                --model "/work/mech-ai-scratch/shreyang/Agentic Ag/distillation/sft_models/phi-3-deepseek-finetuned/checkpoint-12000/merged" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_sft_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_sft_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3__sft_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_sft_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_sft_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/phi3_sft_gbm.json" 

# Phi3 without SFT GBM
python raw_hf.py --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_phi3_gbm.json"\
                --model "microsoft/phi-3-mini-4k-instruct" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_phi3_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_phi3_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_phi3_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/phi3_gbm.json" 

#Qwen-8B SFT GBM
python raw_hf.py  --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_qwen3_sft_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_qwen3_sft_gbm.json"\
                --model "/work/mech-ai-scratch/shreyang/Agentic Ag/distillation/sft_models/qwen3-8b/checkpoint-12000/merged" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_qwen3_sft_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_qwen3_sft_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_qwen3_sft_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_qwen3_sft_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_qwen3_sft_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/qwen3_sft_gbm.json" 

# Qwen-8B without SFT GBM
python raw_hf.py  --input_file "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                --output_file "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_qwen3_gbm.json"\
                --partial_file_path "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/partial/partial_path_qwen3_gbm.json"\
                --model "Qwen/Qwen3-8B" --from_benchmark

python format.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/raw/response_qwen3_gbm.json"\
                 --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_qwen3_gbm.json" 

python merge_metadata.py --input "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/formatted/response_qwen3_gbm.json"\
                         --output "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_qwen3_gbm.json" 

python create_benchmark_response.py --benchmark_json "/work/mech-ai-scratch/shreyang/Agentic Ag/query_generation/benchmark/final_annotation_questions_100.json"\
                                    --larger_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/with_metadata/response_qwen3_gbm.json"\
                                    --output_json "/work/mech-ai-scratch/shreyang/Agentic Ag/response_generation/response/benchmark/qwen3_gbm.json" 