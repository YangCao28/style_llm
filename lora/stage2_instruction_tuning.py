"""LoRA stage-2 (instruction tuning) training script.

从 Stage 1 checkpoint 继续训练，使用对话格式的数据进行指令微调。

Usage:
    python -m lora.stage2_instruction_tuning --config lora/stage2_instruction_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

DEFAULT_SYSTEM_PROMPT = "你是一个文学创作助手，擅长用各种风格改写文本。"


def formatting_func_stage2(example):
    """格式化 stage2 对话数据为训练格式"""
    conversations = example.get("conversations", [])
    if not conversations:
        return {"text": ""}
    
    # 构建对话格式
    text_parts = []
    for msg in conversations:
        role = msg.get("from", "")
        content = msg.get("value", "")
        
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "human":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "gpt":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    return {"text": "\n".join(text_parts)}


class LossRecorderCallback(TrainerCallback):
    """记录训练 loss"""
    def __init__(self):
        self.training_losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_losses.append(logs["loss"])
            self.steps.append(state.global_step)


def main():
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    # 1. 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    
    args = parser.parse_args()
    
    # 加载配置文件
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with args.config.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        # 用配置文件中的值覆盖默认值（命令行参数优先）
        for key, value in config_data.items():
            if not hasattr(args, key) or getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # 检查必需参数
    if not args.model_name_or_path or not args.dataset_path or not args.output_dir:
        parser.error("Required arguments: --model_name_or_path, --dataset_path, --output_dir (or provide via --config)")
        
    args.dataset_path = Path(args.dataset_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Stage 2: Instruction Tuning")
    print("=" * 80)
    print(f"Model checkpoint: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    
    # 2. 加载数据集
    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    print(f"✓ Loaded {len(dataset):,} samples")
    
    # 3. 加载 tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # 4. 加载模型（这是 stage1 checkpoint，已经包含 LoRA）
    print(f"\nLoading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    print(f"✓ Model loaded (继续训练已有的 LoRA 权重)")
    
    # 5. Tokenize 数据集
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        # 当 batched=True 时，examples 是字典，每个键对应一个列表
        texts = []
        num_samples = len(examples["conversations"])
        
        for i in range(num_samples):
            example = {key: examples[key][i] for key in examples}
            formatted = formatting_func_stage2(example)
            texts.append(formatted["text"])
        
        result = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        
        # 设置 labels（用于计算 loss）
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
    )
    print(f"✓ Tokenization complete: {len(tokenized_dataset):,} samples")
    
    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_drop_last=False,
    )
    
    # 7. 创建 data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 8. 创建 Trainer
    loss_recorder = LossRecorderCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[loss_recorder],
    )
    
    # 9
    # 8. 开始训练
    print("\n" + "=" * 80)
    pr10nt("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 9. 保存最终模型
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {args.output_dir}")
    if loss_recorder.training_losses:
        print(f"  Initial loss: {loss_recorder.training_losses[0]:.4f}")
        print(f"  Final loss: {loss_recorder.training_losses[-1]:.4f}")
        print(f"  Total steps: {len(loss_recorder.steps)}")


if __name__ == "__main__":
    main()
