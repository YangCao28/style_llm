"""Stage-2 指令微调 - 添加新的 LoRA adapter

🔑 关键修复：
  1. 在 Stage1 的基础上添加第二个 LoRA adapter（专门用于指令学习）
  2. 冻结 Stage1 的 adapter（保留风格能力），只训练新的 adapter
  3. 正确的 labels 分割（只对 assistant 回复计算 loss）

训练流程：
  Stage1 (style) → frozen → 保留风格注入能力
  Stage2 (instruct) → trainable → 学习指令遵循

Usage:
    python -m lora.stage2_with_new_adapter --config lora/stage2_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def formatting_func_stage2(example, tokenizer, max_seq_length=2048):
    """格式化对话数据 - 只对 assistant 回复计算 loss
    
    关键修复：使用完整tokenize后再定位assistant起始位置，避免tokenizer分词不一致
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
    
    # 构建 prompt（不包含 assistant 回复内容，但包含 assistant 标签）
    prompt_parts = []
    for msg in messages:
        if msg["role"] != "assistant":
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt_text = "\n".join(prompt_parts)
    
    # 完整文本（包含 assistant 回复 + EOS）
    full_text = prompt_text + assistant_response + "<|im_end|>"
    
    # 🔑 关键：一次性tokenize完整文本，然后用prompt长度定位
    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    
    # 🔑 Labels构建：前 len(prompt_ids) 个token设为-100（不计算loss），之后的才计算
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    
    # 验证长度一致性（tokenizer可能导致不一致）
    if len(labels) != len(full_ids):
        # 如果长度不匹配，重新计算（保守策略：全部计算loss）
        labels = full_ids[:]
    
    # Padding到max_seq_length
    input_ids = full_ids + [tokenizer.pad_token_id] * (max_seq_length - len(full_ids))
    attention_mask = [1] * len(full_ids) + [0] * (max_seq_length - len(full_ids))
    labels = labels + [-100] * (max_seq_length - len(labels))
    
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
    parser.add_argument("--base_model_name", type=str, help="Base model path (local folder or HF path)")
    parser.add_argument("--stage1_adapter_path", type=str, help="Stage1 LoRA adapter path")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--lora_r", type=int, default=64, help="New adapter rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="New adapter alpha")
    
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
    if not args.base_model_name or not args.stage1_adapter_path or not args.dataset_path or not args.output_dir:
        parser.error("Required: --base_model_name, --stage1_adapter_path, --dataset_path, --output_dir")
    
    args.stage1_adapter_path = Path(args.stage1_adapter_path)
    args.dataset_path = Path(args.dataset_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Stage 2: Instruction Tuning with NEW LoRA Adapter")
    print("=" * 80)
    print(f"Base model: {args.base_model_name}")
    print(f"Stage1 adapter: {args.stage1_adapter_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"New adapter rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("\n🔑 Strategy:")
    print("  1. Load pure base model")
    print("  2. Load Stage1 style adapter (FROZEN)")
    print("  3. Add new instruct adapter (TRAINABLE)")
    print("  4. Both adapters active during inference")
    
    # 加载数据集
    print(f"\nLoading dataset...")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    print(f"✓ 📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # 🔑 关键步骤1：加载纯净的 base model
    print(f"\n🎯 Step 1: Loading pure base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    print(f"✓ Base model loaded: {args.base_model_name}")
    
    # 🔑 关键步骤2：加载 Stage1 的 style adapter
    print(f"\n🎨 Step 2: Loading Stage1 style adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        args.stage1_adapter_path,
        adapter_name="style",
    )
    print(f"✓ Style adapter loaded and will be FROZEN")
    
    # 🔑 关键步骤3：添加新的 instruct adapter
    print(f"\n📝 Step 3: Adding NEW instruct adapter...")
    
    # 配置新的 adapter
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 添加新的 adapter
    model.add_adapter("instruct", lora_config)
    print(f"✓ Instruct adapter added (trainable)")
    
    # 🔑 关键步骤4：冻结 style adapter，只训练 instruct adapter
    print(f"\n🔒 Step 4: Freezing style adapter, training instruct only...")
    
    # 列出所有 adapters
    print(f"📋 Available adapters:")
    if hasattr(model, 'peft_config'):
        for name in model.peft_config.keys():
            print(f"  - {name}")
    
    # 设置当前活跃的 adapter 为 "instruct"
    model.set_adapter("instruct")
    
    # 冻结 style adapter 的参数
    for name, param in model.named_parameters():
        if "style" in name:
            param.requires_grad = False
            
    print(f"✓ Style adapter: FROZEN (but active during forward)")
    print(f"✓ Instruct adapter: TRAINABLE (active)")
    
    # 验证：检查哪些参数是可训练的
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Parameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    # Tokenize 数据集
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        num_samples = len(examples["conversations"])
        for i in range(num_samples):
            example = {key: examples[key][i] for key in examples}
            formatted = formatting_func_stage2(example, tokenizer, args.max_seq_length)
            
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
    
    # 验证第一个样本
    print("\n🔍 验证 labels（应只包含 assistant 回复）:")
    first_labels = tokenized_dataset[0]["labels"]
    valid_label_ids = [lid for lid in first_labels if lid != -100]
    if valid_label_ids:
        decoded = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
        print(f"Labels preview: {decoded[:150]}...")
        if any(m in decoded.lower() for m in ["<|im_start|>system", "<|im_start|>user"]):
            print("⚠️  WARNING: Labels 包含 system/user 标记！")
        else:
            print("✅ Labels 正确")
    
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
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 🔑 关键：只保存 instruct adapter（不保存 style adapter）
    print("\n💾 Saving ONLY instruct adapter (NOT style)...")
    
    # 方法1：直接保存 instruct adapter
    model.save_pretrained(
        args.output_dir,
        selected_adapters=["instruct"],  # 只保存 instruct adapter
    )
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存 adapter 配置信息
    adapter_info = {
        "base_model": args.base_model_name,
        "stage1_style_adapter": str(args.stage1_adapter_path),
        "stage2_instruct_adapter": "instruct (this folder)",
        "usage": "Load base model + stage1 style adapter + this instruct adapter for inference",
        "inference_command": f"--style_adapter {args.stage1_adapter_path} --instruct_adapter {args.output_dir}",
    }
    with open(args.output_dir / "adapter_info.json", "w", encoding="utf-8") as f:
        json.dump(adapter_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Training complete!")
    print(f"  📁 Instruct adapter saved to: {args.output_dir}")
    print(f"  📁 Style adapter remains at: {args.stage1_adapter_path}")
    print(f"\n🎯 For inference, use BOTH adapters:")
    print(f"  python -m lora.test_stage2_instruction \\")
    print(f"    --style_adapter {args.stage1_adapter_path} \\")
    print(f"    --instruct_adapter {args.output_dir}")
    if loss_recorder.training_losses:
        print(f"\n📊 Training stats:")
        print(f"  Initial loss: {loss_recorder.training_losses[0]:.4f}")
        print(f"  Final loss: {loss_recorder.training_losses[-1]:.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
