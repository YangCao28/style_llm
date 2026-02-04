"""单阶段指令微调 - 直接在 Base Model 上训练

🔑 关键改进：
  1. 跳过 Stage1，直接在 base model 上训练指令能力
  2. 避免续写数据的干扰
  3. 更高的学习率和更多训练轮次
  4. 正确的 labels 分割（只对 assistant 回复计算 loss）

Usage:
    python -m lora.single_stage_instruct --config lora/single_stage_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


def formatting_func_instruct(example, tokenizer, max_seq_length=2048, debug=False):
    """格式化对话数据 - 只对 assistant 回复计算 loss
    
    正确方法：分别tokenize prompt和assistant，然后拼接并构建labels
    """
    conversations = example.get("conversations", [])
    if not conversations:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 构建对话
    messages = []
    assistant_response = None
    
    for msg in conversations:
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        
        if role in ("system", "sys"):
            norm_role = "system"
        elif role in ("assistant", "gpt", "bot"):
            norm_role = "assistant"
            assistant_response = content.strip()
        else:
            norm_role = "user"
        
        messages.append({"role": norm_role, "content": content})
    
    if assistant_response is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 构建 prompt（不包含 assistant 回复内容，但包含 assistant 开始标签）
    prompt_parts = []
    for msg in messages:
        if msg["role"] != "assistant":
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt_text = "\n".join(prompt_parts)
    
    # 🔑 关键修复：分别tokenize，避免BPE分词不一致
    # Tokenize prompt部分（不计算loss）
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    
    # Tokenize assistant回复部分（计算loss）+ EOS标记
    assistant_text = assistant_response + "<|im_end|>"
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    
    # 拼接完整序列
    input_ids = prompt_ids + assistant_ids
    
    # 截断到max_seq_length
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        # 确保prompt不被截断（如果prompt太长，只能截断assistant）
        if len(prompt_ids) > max_seq_length:
            # Prompt太长，跳过这个样本
            return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 构建labels：prompt部分为-100，assistant部分为实际token ids
    labels = [-100] * len(prompt_ids) + assistant_ids
    labels = labels[:max_seq_length]  # 截断到max_seq_length
    
    # Padding到max_seq_length
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * len(input_ids[:max_seq_length - padding_length]) + [0] * padding_length
    labels = labels + [-100] * (max_seq_length - len(labels))
    
    # Debug打印第一个样本
    if debug:
        print("\n" + "="*80)
        print("🔍 DEBUG: formatting_func_instruct 第一个样本详情")
        print("="*80)
        print(f"\n📝 Prompt文本 ({len(prompt_ids)} tokens):")
        print(prompt_text[:200] + "...")
        print(f"\n✅ Assistant文本 ({len(assistant_ids)} tokens):")
        print(assistant_text[:200] + "...")
        print(f"\n📊 统计:")
        print(f"  Prompt tokens: {len(prompt_ids)}")
        print(f"  Assistant tokens: {len(assistant_ids)}")
        print(f"  Total tokens: {len(input_ids[:max_seq_length-padding_length])}")
        print(f"  Labels中-100数量: {sum(1 for l in labels if l == -100)}")
        print(f"  Labels中有效数量: {sum(1 for l in labels if l != -100)}")
        print("="*80 + "\n")
    
    return {
        "input_ids": input_ids[:max_seq_length],
        "attention_mask": attention_mask[:max_seq_length],
        "labels": labels[:max_seq_length],
    }


class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.training_losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_losses.append(logs["loss"])
            self.steps.append(state.global_step)
            if getattr(state, "is_world_process_zero", True):
                print(f"[step {state.global_step}] loss = {logs['loss']:.4f}")


def main():
    # CUDA 初始化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--base_model_name", type=str, help="Base model path")
    parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path (if different from base_model_name)")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha")
    
    args = parser.parse_args()
    
    # 加载配置文件
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with args.config.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            if not hasattr(args, key) or getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # 检查必需参数
    if not args.base_model_name or not args.dataset_path or not args.output_dir:
        parser.error("Required: --base_model_name, --dataset_path, --output_dir")
    
    args.dataset_path = Path(args.dataset_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Single Stage: Instruction Fine-tuning on Base Model")
    print("=" * 80)
    print(f"Base model: {args.base_model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    
    # 加载数据集
    print(f"\n📚 Loading dataset...")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    print(f"\n📚 Loading tokenizer...")
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # 加载 base model
    print(f"\n🎯 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    )
    print(f"✓ Base model loaded: {args.base_model_name}")
    
    # 配置 LoRA
    print(f"\n📝 Adding LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, lora_config)
    print(f"✓ LoRA adapter added")
    
    # 验证：检查哪些参数是可训练的
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Parameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    # Tokenize 数据集
    print("\n🔄 Tokenizing dataset...")
    
    # 先处理第一个样本用于debug
    print("\n" + "="*80)
    print("🔍 处理第一个样本 (debug mode)")
    print("="*80)
    formatting_func_instruct(dataset[0], tokenizer, args.max_seq_length, debug=True)
    
    def tokenize_function(examples):
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        num_samples = len(examples["conversations"])
        for i in range(num_samples):
            example = {key: examples[key][i] for key in examples}
            formatted = formatting_func_instruct(example, tokenizer, args.max_seq_length)
            
            if formatted["input_ids"]:
                results["input_ids"].append(formatted["input_ids"])
                results["attention_mask"].append(formatted["attention_mask"])
                results["labels"].append(formatted["labels"])
        
        return results
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )
    print(f"✓ Tokenization complete: {len(tokenized_dataset):,} samples")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 训练参数
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
    )
    
    # 创建 Trainer
    loss_recorder = LossRecorderCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[loss_recorder],
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🚀 Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 保存模型
    print("\n💾 Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存配置信息
    model_info = {
        "base_model": args.base_model_name,
        "training_type": "single_stage_instruction_tuning",
        "dataset": str(args.dataset_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
    }
    with open(args.output_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Training complete!")
    print(f"  📁 Model saved to: {args.output_dir}")
    
    if loss_recorder.training_losses:
        print(f"\n📊 Training stats:")
        print(f"  Initial loss: {loss_recorder.training_losses[0]:.4f}")
        print(f"  Final loss: {loss_recorder.training_losses[-1]:.4f}")
        print(f"  Loss reduction: {loss_recorder.training_losses[0] - loss_recorder.training_losses[-1]:.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
