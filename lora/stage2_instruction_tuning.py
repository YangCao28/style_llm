"""LoRA stage-2 (instruction tuning) training script with FIXED label segmentation.

🔧 关键修复：
  - Labels 只包含 assistant 的回复文本，不包含 system/user/assistant 标记
  - 这样模型就不会学到"继续对话"的行为

从 Stage 1 checkpoint 继续训练，使用对话格式的数据进行指令微调。

Usage:
    python -m lora.stage2_instruction_tuning_fixed --config lora/stage2_instruction_config.json
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
)

DEFAULT_SYSTEM_PROMPT = "你是一个文学创作助手，擅长用各种风格改写文本。"


def formatting_func_stage2_fixed(example, tokenizer, max_seq_length=2048):
    """格式化 stage2 对话数据 - 关键修复：labels 只包含 assistant 回复
    
    ✅ 正确做法：
      - input_ids: 包含完整对话（system + user + assistant 开头）
      - labels: 只标注 assistant 的回复文本 + EOS，其他部分设为 -100（忽略）
      - 强制在 assistant 回复结尾添加 <|im_end|> 作为明确的停止信号
    
    ❌ 错误做法（旧版）：
      - 整个文本都作为 label，导致模型学到 "继续对话" 的行为
      - 没有强制 EOS token，导致模型不知道何时停止
    """
    conversations = example.get("conversations", [])
    if not conversations:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 构建完整的 prompt 和找到 assistant 的回复
    messages = []
    assistant_response = None
    
    for msg in conversations:
        # 兼容两种字段命名
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        
        # 归一化角色名称
        if role in ("system", "sys"):
            norm_role = "system"
        elif role in ("assistant", "gpt", "bot"):
            norm_role = "assistant"
            assistant_response = content  # 保存 assistant 的回复
        else:
            norm_role = "user"
        
        messages.append({"role": norm_role, "content": content})
    
    if assistant_response is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 确保 assistant 回复不包含额外的后缀（如"改写完成"、"请参考"等）
    # 清理可能的尾巴
    assistant_response = assistant_response.strip()
    
    # 使用 apply_chat_template 构建完整对话
    # 注意：我们需要先构建不包含 assistant 回复的 prompt，然后再加上 assistant 的部分
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    
    # 手动构建（因为 Qwen 的 tokenizer 可能不支持 apply_chat_template）
    prompt_parts = []
    for msg in prompt_messages:
        prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt_text = "\n".join(prompt_parts)
    
    # 完整文本（包含 assistant 回复 + 强制的结束标记）
    # 🔑 关键：确保 <|im_end|> 被包含在训练中，让模型学会"说完就停"
    full_text = prompt_text + assistant_response + "<|im_end|>"
    
    # Tokenize
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
    
    # 构建 labels：只有 assistant 回复部分（包括 <|im_end|>）是有效的，其他部分设为 -100
    # 这样模型会学到：生成回复内容 → 输出 <|im_end|> → 停止
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    # Padding to max_length
    input_ids = full_ids + [tokenizer.pad_token_id] * (max_seq_length - len(full_ids))
    attention_mask = [1] * len(full_ids) + [0] * (max_seq_length - len(full_ids))
    labels = labels + [-100] * (max_seq_length - len(labels))
    
    return {
        "input_ids": input_ids[:max_seq_length],
        "attention_mask": attention_mask[:max_seq_length],
        "labels": labels[:max_seq_length],
    }


class LossRecorderCallback(TrainerCallback):
    """记录训练 loss"""
    def __init__(self):
        self.training_losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_losses.append(logs["loss"])
            self.steps.append(state.global_step)
            # 打印当前 step 的 loss，频率由 TrainingArguments.logging_steps 控制
            if getattr(state, "is_world_process_zero", True):
                print(f"[step {state.global_step}] loss = {logs['loss']:.4f}")


def main():
    # 强制清理 CUDA 缓存和重置设备
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 尝试初始化 CUDA
        try:
            _ = torch.zeros(1).cuda()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"✓ CUDA initialized: {torch.cuda.get_device_name(0)}")
            print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except RuntimeError as e:
            print(f"⚠️  CUDA initialization failed: {e}")
            print("  Try: pkill -9 python; nvidia-smi --gpu-reset")
            raise
    
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
    parser.add_argument("--save_steps", type=int, default=50)
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
    print("Stage 2: Instruction Tuning (FIXED - Proper Label Segmentation)")
    print("=" * 80)
    print(f"Model checkpoint: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print("\n🔧 Key Fix: Labels only contain assistant response (no role markers)")
    
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
    
    # 5. Tokenize 数据集 - 使用修复后的格式化函数
    print("\nTokenizing dataset with proper label segmentation...")
    def tokenize_function(examples):
        # 当 batched=True 时，examples 是字典，每个键对应一个列表
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        num_samples = len(examples["conversations"])
        for i in range(num_samples):
            example = {key: examples[key][i] for key in examples}
            formatted = formatting_func_stage2_fixed(example, tokenizer, args.max_seq_length)
            
            if formatted["input_ids"]:  # 只添加有效样本
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
    
    # 验证：打印第一个样本的 labels，确保没有 role markers
    print("\n🔍 验证第一个样本的 labels (应该只包含 assistant 回复):")
    first_labels = tokenized_dataset[0]["labels"]
    # 找到第一个非 -100 的位置
    valid_label_ids = [lid for lid in first_labels if lid != -100]
    if valid_label_ids:
        decoded_labels = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
        print(f"Labels preview (前200字符): {decoded_labels[:200]}")
        if any(marker in decoded_labels.lower() for marker in ["<|im_start|>", "system", "user", "assistant"]):
            print("⚠️  WARNING: Labels 包含 role markers！这会导致模型继续对话。")
        else:
            print("✅ Labels 看起来正确（只有回复内容）")
    
    # 清理一次显存（仅在有 CUDA 时）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
    # 7. 创建 Trainer（不需要 data_collator，因为我们已经做好了 padding）
    loss_recorder = LossRecorderCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[loss_recorder],
    )
    
    # 8. 开始训练
    print("\n" + "=" * 80)
    print("Starting training...")
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

    # 训练结束后再清一次显存，方便同一进程后续继续使用 GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
