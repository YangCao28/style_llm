"""Stage2 指令微调一键训练脚本

整合了数据准备、清理和训练的完整流程：
1. 从完整数据集中采样子集（可选）
2. 清理 assistant 回复中的"尾巴"
3. 执行 Stage2 训练

Usage:
    # 使用完整数据集训练
    python train_stage2_complete.py --config lora/stage2_instruction_config.json
    
    # 使用子集训练（轻量校正）
    python train_stage2_complete.py \
        --config lora/stage2_correction_config.json \
        --use_subset \
        --subset_ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
import torch.nn.functional as F


class WeightedLossTrainer(Trainer):
    """支持 per-token loss 权重的自定义 Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算带权重的 loss"""
        # 提取权重（如果存在）
        loss_weight = inputs.pop("loss_weight", None)
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # 计算 loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算每个 token 的 loss（不做 reduction）
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 只对非 -100 的 token 计算 loss
        mask = (shift_labels.view(-1) != -100).float()
        
        # 应用权重：EOS token 贡献更多，但分母不变保持 loss 量级
        if loss_weight is not None:
            shift_weight = loss_weight[..., 1:].contiguous().view(-1)
            # 加权loss总和 / 有效token数量（不是加权的token数量）
            loss = (loss * shift_weight * mask).sum() / mask.sum()
        else:
            loss = (loss * mask).sum() / mask.sum()
        
        return (loss, outputs) if return_outputs else loss

DEFAULT_SYSTEM_PROMPT = "你是一个文学创作助手，擅长用各种风格改写文本。"

# 需要清理的常见后缀模式
TAIL_PATTERNS = [
    r"改写完成[。！\.!]*$",
    r"请参考[。！\.!]*$",
    r"以上是改写结果[。！\.!]*$",
    r"改写如下[：:。！\.!]*$",
    r"供您参考[。！\.!]*$",
    r"希望对您有帮助[。！\.!]*$",
    r"谢谢[。！\.!]*$",
    r"还有其他问题吗[？?。！\.!]*$",
    r"[\n\s]+$",  # 结尾的多余空白
]


def clean_assistant_response(response: str) -> str:
    """清理 assistant 回复中的尾巴"""
    cleaned = response.strip()
    
    # 应用所有清理模式
    for pattern in TAIL_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()


def prepare_dataset(
    input_path: Path,
    use_subset: bool = False,
    subset_ratio: float = 0.2,
    seed: int = 42,
) -> List[Dict]:
    """准备训练数据集：加载、采样、清理"""
    print(f"\n{'='*80}")
    print("📖 准备训练数据")
    print(f"{'='*80}")
    print(f"输入路径: {input_path}")
    
    # 1. 加载数据
    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"✓ 加载了 {len(samples):,} 个样本")
    
    # 2. 采样（如果需要）
    if use_subset and subset_ratio < 1.0:
        random.seed(seed)
        num_samples = int(len(samples) * subset_ratio)
        samples = random.sample(samples, num_samples)
        print(f"✓ 采样了 {num_samples:,} 个样本 ({subset_ratio:.1%})")
    
    # 3. 清理 assistant 回复
    cleaned_count = 0
    for sample in samples:
        conversations = sample.get("conversations", [])
        for msg in conversations:
            role = msg.get("role") or msg.get("from")
            if role in ("assistant", "gpt", "bot"):
                content = msg.get("content") or msg.get("value") or ""
                cleaned_content = clean_assistant_response(content)
                
                if cleaned_content != content:
                    cleaned_count += 1
                    if "content" in msg:
                        msg["content"] = cleaned_content
                    if "value" in msg:
                        msg["value"] = cleaned_content
    
    print(f"✓ 清理了 {cleaned_count} 个 assistant 回复中的'尾巴'")
    
    return samples


def formatting_func_stage2_fixed(example, tokenizer, max_seq_length=2048, eos_loss_weight=2.0):
    """格式化 stage2 对话数据 - labels 只包含 assistant 回复 + EOS
    
    Args:
        eos_loss_weight: EOS token 的 loss 权重倍数。>1.0 会强化停止行为的学习。
    """
    conversations = example.get("conversations", [])
    if not conversations:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 构建完整的 prompt 和找到 assistant 的回复
    messages = []
    assistant_response = None
    
    for msg in conversations:
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        
        if role in ("system", "sys"):
            norm_role = "system"
        elif role in ("assistant", "gpt", "bot"):
            norm_role = "assistant"
            assistant_response = content
        else:
            norm_role = "user"
        
        messages.append({"role": norm_role, "content": content})
    
    if assistant_response is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 确保 assistant 回复已清理
    assistant_response = assistant_response.strip()
    
    # 构建 prompt（不包含 assistant 回复）
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_parts = []
    for msg in prompt_messages:
        prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt_text = "\n".join(prompt_parts)
    
    # 完整文本（包含 assistant 回复 + 强制的结束标记）
    full_text = prompt_text + assistant_response + "<|im_end|>"
    
    # Tokenize
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
    
    # 获取 EOS token id
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # 构建 labels：只有 assistant 回复部分（包括 <|im_end|>）是有效的
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    # 🔑 为 EOS token 创建权重（如果需要强化学习）
    loss_weight = [1.0] * len(labels)  # 默认权重都是 1.0
    if eos_loss_weight > 1.0:
        for i, label_id in enumerate(labels):
            if label_id == eos_token_id:
                loss_weight[i] = eos_loss_weight
    
    # Padding
    input_ids = full_ids + [tokenizer.pad_token_id] * (max_seq_length - len(full_ids))
    attention_mask = [1] * len(full_ids) + [0] * (max_seq_length - len(full_ids))
    labels = labels + [-100] * (max_seq_length - len(labels))
    loss_weight = loss_weight + [0.0] * (max_seq_length - len(loss_weight))
    
    return {
        "input_ids": input_ids[:max_seq_length],
        "attention_mask": attention_mask[:max_seq_length],
        "labels": labels[:max_seq_length],
        "loss_weight": loss_weight[:max_seq_length],  # 新增：每个 token 的权重
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
            if getattr(state, "is_world_process_zero", True):
                print(f"[step {state.global_step}] loss = {logs['loss']:.4f}")


def main():
    # 强制清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _ = torch.zeros(1).cuda()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"✓ CUDA initialized: {torch.cuda.get_device_name(0)}")
            print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except RuntimeError as e:
            print(f"⚠️  CUDA initialization failed: {e}")
            raise
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file")
    parser.add_argument("--use_subset", action="store_true", help="Use subset of data for training")
    parser.add_argument("--subset_ratio", type=float, default=0.2, help="Ratio of data to use (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling")
    
    # 配置参数（可以被配置文件覆盖）
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
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank (只在从 base model 训练时生效)")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha (只在从 base model 训练时生效)")
    parser.add_argument("--eos_loss_weight", type=float, default=2.0, help="EOS token 的 loss 权重倍数，>1.0 会强化停止行为学习")
    
    args = parser.parse_args()
    
    # 加载配置文件
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with args.config.open("r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # 用配置文件中的值覆盖默认值
    for key, value in config_data.items():
        if not hasattr(args, key) or getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    
    # 检查必需参数
    if not args.model_name_or_path or not args.dataset_path or not args.output_dir:
        parser.error("Required arguments: --model_name_or_path, --dataset_path, --output_dir (or provide via --config)")
    
    args.dataset_path = Path(args.dataset_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU 缓存已清理")
    
    print("=" * 80)
    print("Stage 2: 一键式指令微调训练")
    print("=" * 80)
    print(f"Model checkpoint: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Max seq length: {args.max_seq_length}")
    if args.use_subset:
        print(f"Using subset: {args.subset_ratio:.1%} of data")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    
    # 1. 准备数据集（加载、采样、清理）
    samples = prepare_dataset(
        args.dataset_path,
        use_subset=args.use_subset,
        subset_ratio=args.subset_ratio,
        seed=args.seed,
    )
    
    # 保存到临时文件
    temp_data_path = args.output_dir / "training_data_cleaned.jsonl"
    with temp_data_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✓ 保存清理后的数据到: {temp_data_path}")
    
    # 2. 加载 tokenizer
    print(f"\n{'='*80}")
    print("🔧 加载模型和 Tokenizer")
    print(f"{'='*80}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    print(f"✓ Model loaded")
    
    # 4. 加载并处理数据集
    print(f"\n{'='*80}")
    print("📊 Tokenizing dataset")
    print(f"{'='*80}")
    dataset = load_dataset("json", data_files=str(temp_data_path), split="train")
    
    def tokenize_function(examples):
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weight": [],
        }
        
        num_samples = len(examples["conversations"])
        for i in range(num_samples):
            example = {key: examples[key][i] for key in examples}
            formatted = formatting_func_stage2_fixed(example, tokenizer, args.max_seq_length, args.eos_loss_weight)
            
            if formatted["input_ids"]:
                results["input_ids"].append(formatted["input_ids"])
                results["attention_mask"].append(formatted["attention_mask"])
                results["labels"].append(formatted["labels"])
                results["loss_weight"].append(formatted["loss_weight"])
        
        return results
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )
    print(f"✓ Tokenization complete: {len(tokenized_dataset):,} samples")
    
    # 验证第一个样本
    print("\n🔍 验证第一个样本的 labels:")
    first_labels = tokenized_dataset[0]["labels"]
    valid_label_ids = [lid for lid in first_labels if lid != -100]
    if valid_label_ids:
        decoded_labels = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
        print(f"Labels preview (前200字符): {decoded_labels[:200]}")
        if any(marker in decoded_labels.lower() for marker in ["<|im_start|>", "system", "user"]):
            print("⚠️  WARNING: Labels 包含 role markers！")
        else:
            print("✅ Labels 正确（只有回复内容 + <|im_end|>）")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 5. 训练参数
    print(f"\n{'='*80}")
    print("⚙️  配置训练参数")
    print(f"{'='*80}")
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
    
    # 6. 创建 Trainer（使用自定义的 WeightedLossTrainer）
    loss_recorder = LossRecorderCallback()
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[loss_recorder],
    )
    
    # 7. 开始训练
    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 8. 保存最终模型
    print("\n💾 保存最终模型...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✓ 训练完成!")
    print(f"  Model saved to: {args.output_dir}")
    if loss_recorder.training_losses:
        print(f"  Initial loss: {loss_recorder.training_losses[0]:.4f}")
        print(f"  Final loss: {loss_recorder.training_losses[-1]:.4f}")
        print(f"  Total steps: {len(loss_recorder.steps)}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
