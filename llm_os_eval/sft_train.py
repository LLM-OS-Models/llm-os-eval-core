#!/usr/bin/env python3
"""Shared SFT training script using LoRA + trl.

Usage:
    python sft_train.py \
        --base-model Qwen/Qwen3.5-4B \
        --data-dir data/sft \
        --output-dir checkpoints/sft_run1 \
        --gpu 0 \
        --epochs 3 \
        --batch-size 4 \
        --lr 2e-5 \
        --lora-r 16
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset


def load_sft_data(data_dir: str) -> Dataset:
    train_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    def gen():
        for r in rows:
            yield {
                "messages": r.get("messages", []),
                "text": r.get("text", ""),
            }

    return Dataset.from_generator(gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="HuggingFace model ID")
    parser.add_argument("--data-dir", required=True, help="Directory with train.jsonl/val.jsonl")
    parser.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data from {args.data_dir}")
    train_dataset = load_sft_data(args.data_dir)

    val_dataset = None
    val_path = os.path.join(args.data_dir, "val.jsonl")
    if os.path.exists(val_path):
        val_rows = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_rows.append(json.loads(line))

        def val_gen():
            for r in val_rows:
                yield {"messages": r.get("messages", []), "text": r.get("text", "")}

        val_dataset = Dataset.from_generator(val_gen)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        print(f"Train: {len(train_dataset)}, no validation set")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        max_seq_length=args.max_seq_length,
        report_to="none",
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
