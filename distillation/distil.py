import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer
import copy

class MergingSFTTrainer(SFTTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)  # Save checkpoint (with LoRA)

        # Clone model before merging
        if hasattr(self.model, "merge_and_unload"):
            print(f"[Merging and saving model to {output_dir}/merged]")
            try:
                merged_model = copy.deepcopy(self.model).merge_and_unload()
                merged_dir = os.path.join(output_dir, "merged")
                merged_model.save_pretrained(merged_dir)
                self.tokenizer.save_pretrained(merged_dir)
            except Exception as e:
                print(f"[Warning] Could not save merged model: {e}")



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with filtered data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data JSON file")
    parser.add_argument("--benchmark_path", type=str, default="./final_annotation_questions.json", help="Path to benchmark questions JSON")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID to use")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=1200, help="Steps interval for evaluation and logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    with open(args.data_path) as f:
        data = json.load(f)

    with open(args.benchmark_path) as f:
        benchmark_data = json.load(f)

    benchmark_questions = [q["processed_question"] for q in benchmark_data]
    data = [qa for qa in data if qa["question"] not in benchmark_questions]

    dataset = Dataset.from_list(data)

    def format_instruction(example):
        return {
            "text": (
                "<|user|>\n"
                f"{example['question']}\n"
                "<|end|>\n"
                "<|assistant|>\n"
                f"{example['response']}\n"
                "<|end|>"
            )
        }

    formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=dataset.column_names)
    formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    CUSTOM_TOKENS = ["<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_id=="Qwen/Qwen3-8B":
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="best",
        metric_for_best_model="loss",
        logging_steps=args.eval_steps,
        logging_dir=args.output_dir,
        learning_rate=2e-5,
        fp16=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = MergingSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        data_collator=data_collator,
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(f"{args.output_dir}-final")
    tokenizer.save_pretrained(f"{args.output_dir}-final")


if __name__ == "__main__":
    main()
