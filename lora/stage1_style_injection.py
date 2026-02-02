"""LoRA stage-1 (style injection) training script for 45k x 1024-token corpus.

训练流程说明：
1. 加载原始JSONL数据（每条记录包含"text"字段）
2. 对每条文本进行格式化：添加提示模板（"### 文风语料\n【packed_style_sample】\n### 正文\n"）+ 正文内容
3. Tokenize所有文本，不padding
4. 在DataLoader阶段使用PackingDataCollator：
   - 从batch中的多个样本拼接token ids
   - 每个样本之间插入EOS token作为分隔
   - 打包成4096 token的完整序列
   - 动态生成attention_mask和labels
5. 使用LoRA训练（r=128, alpha=256），只训练q/k/v/o/gate/up/down投影层
6. BF16混合精度 + gradient checkpointing节省显存

效率优势：
- Packing避免padding浪费，GPU利用率接近100%
- LoRA只训练~2%参数，显存占用低
- 单卡A100/A800可以跑batch_size=8 * grad_accum=16 = 有效batch 128

Usage:
    python -m lora.stage1_style_injection --config lora/stage1_style_config.example.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

DEFAULT_MODEL = "Qwen/Qwen3-8B-Base"
RESPONSE_TEMPLATE = "### 正文\n"
PROMPT_TEMPLATE = "### 文风语料\n"
MERGED_TOKEN_STATEMENT = "【packed_style_sample】\n"

# 清理文本的正则表达式
META_PATTERNS = [
    re.compile(r"作者[:：].*?$", re.MULTILINE),
    re.compile(r"^\s*第[一二三四五六七八九十百千0-9]+[章回节].*?$", re.MULTILINE),
    re.compile(r"https?://\S+", re.IGNORECASE),
]
WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Stage-1 LoRA style injection training",
        parents=[config_parser],
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/dataset/combined_dataset.jsonl"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./stage1_style_injection"))
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    # LoRA 相关超参，便于后续做 Alpha 增强训练
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    # 续跑用的 checkpoint 路径（可在 JSON config 里指定）
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    if config_args.config:
        if not config_args.config.exists():
            raise FileNotFoundError(config_args.config)
        with config_args.config.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        normalized_defaults: Dict[str, object] = {}
        for key, raw_value in config_data.items():
            matched_action = next((a for a in parser._actions if a.dest == key), None)
            if matched_action is None:
                raise ValueError(f"Unknown config key '{key}'")
            converter = matched_action.type
            if converter and raw_value is not None and not isinstance(raw_value, converter):
                normalized_defaults[key] = converter(raw_value)
            else:
                normalized_defaults[key] = raw_value
        parser.set_defaults(**normalized_defaults)

    args = parser.parse_args(remaining_argv)
    return args


def normalize_text(record: Dict[str, str]) -> str:
    """清理文本：去除章节标题、作者信息、URL等元数据"""
    text = record.get("text", "")
    for pattern in META_PATTERNS:
        text = pattern.sub("", text)
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    if not text:
        return ""
    paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n\n".join(paragraphs)


def formatting_func(record: Dict[str, str]) -> str:
    """格式化为训练模板：提示词 + 正文"""
    body = normalize_text(record)
    if not body:
        return ""
    return f"{PROMPT_TEMPLATE}{MERGED_TOKEN_STATEMENT}{RESPONSE_TEMPLATE}{body}"


class PackingDataCollator:
    """
    动态打包数据collator：
    - 将batch中的多个样本的token ids拼接起来
    - 每个样本之间插入EOS token
    - 打包成max_length长度的序列
    - 自动生成attention_mask和labels（padding位置label=-100）
    """
    
    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(self, examples):
        # 拼接batch中所有样本的token ids
        all_input_ids = []
        for example in examples:
            ids = example["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            all_input_ids.extend(ids)
            all_input_ids.append(self.eos_token_id)  # 样本间插入EOS
        
        # 切分成max_length的chunk
        packed_sequences = []
        for i in range(0, len(all_input_ids), self.max_length):
            chunk = all_input_ids[i : i + self.max_length]
            if len(chunk) == self.max_length:
                packed_sequences.append(chunk)
            elif chunk:  # 最后一个不完整的chunk
                chunk = chunk + [self.pad_token_id] * (self.max_length - len(chunk))
                packed_sequences.append(chunk)
        
        if not packed_sequences:
            # Fallback：至少返回一个序列
            packed_sequences = [[self.pad_token_id] * self.max_length]
        
        # 转tensor
        input_ids = torch.tensor(packed_sequences, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # padding不参与loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class LossRecorderCallback(TrainerCallback):
    """记录训练过程中的 loss 值"""
    
    def __init__(self):
        self.training_losses: List[float] = []
        self.steps: List[int] = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_losses.append(logs["loss"])
            self.steps.append(state.global_step)


def plot_loss_curve(losses: List[float], steps: List[int], output_dir: Path) -> None:
    """绘制并保存 loss 曲线图"""
    if not losses:
        print("⚠ 没有 loss 数据，跳过绘图")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.8)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    loss_plot_path = output_dir / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss 曲线图已保存到: {loss_plot_path}")
    
    # 保存 loss 数据到 JSON
    loss_data = {
        "steps": steps,
        "losses": losses,
        "min_loss": min(losses),
        "final_loss": losses[-1],
        "total_steps": len(steps)
    }
    loss_json_path = output_dir / "loss_history.json"
    with open(loss_json_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, indent=2)
    print(f"✓ Loss 数据已保存到: {loss_json_path}")
    
    plt.close()



def main() -> None:
    args = parse_args()
    
    # 清理GPU缓存，释放残留显存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ GPU缓存已清理")
    
    # 1. 加载数据集
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(
        "json",
        data_files=str(args.dataset_path),
        split="train",
        streaming=args.streaming,
    )
    
    # 统计数据集大小
    if not args.streaming:
        dataset_size = len(dataset)
        print(f"✓ 数据集加载完成：共 {dataset_size:,} 条训练样本")
    else:
        print(f"✓ 数据集加载完成：流式模式（无法提前统计总数）")
    
    # 2. 加载tokenizer
    print(f"\nLoading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer加载完成")
    
    # 3. 加载模型
    print(f"\nLoading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_impl,
    )
    print(f"✓ 模型加载完成")
    
    # 4. 应用LoRA
    print("\nApplying LoRA configuration")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Tokenize数据集（不padding，保持原始长度）
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        texts = [formatting_func({"text": text}) for text in examples["text"]]
        return tokenizer(
            texts,
            Loss 记录器和 Trainer
    loss_recorder = LossRecorderCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[loss_recorder]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # 打印tokenize后的统计信息
    if not args.streaming:
        print(f"✓ Tokenization完成：{len(tokenized_dataset):,} 条样本已转换为token ids")
        estimated_steps = len(tokenized_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        print(f"  预计训练步数：{estimated_steps} steps")
    else:
        print(f"✓ Tokenization完成（流式模式）")
    
    # 6. 创建packing collator
    data_collator = PackingDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    
    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
    
    # 11. 绘制并保存 loss 曲线
    print("\n生成 Loss 曲线图...")
    plot_loss_curve(loss_recorder.training_losses, loss_recorder.steps, args.output_dir)
    print(f"\n📊 训练摘要:")
    if loss_recorder.training_losses:
        print(f"  最小 Loss: {min(loss_recorder.training_losses):.4f}")
        print(f"  最终 Loss: {loss_recorder.training_losses[-1]:.4f}")
        print(f"  总训练步数: {len(loss_recorder.steps)}")
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )
    
    # 8. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 9. 开始训练（如提供 resume_from_checkpoint，则从对应 checkpoint 续跑）
    if not args.streaming:
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print("\n" + "="*70)
        print(f"开始训练 Stage-1 风格注入")
        print(f"  训练数据：{len(tokenized_dataset):,} 条样本")
        print(f"  Batch size：{args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {effective_batch}")
        print(f"  学习率：{args.learning_rate}")
        print(f"  训练轮数：{args.num_train_epochs}")
        print(f"  预计步数：~{len(tokenized_dataset) // effective_batch} steps")
        print(f"  保存间隔：每 {args.save_steps} steps")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print(f"开始训练 Stage-1 风格注入（流式模式）")
        print("="*70 + "\n")
    
    if args.resume_from_checkpoint:
        print(f"从 checkpoint 续跑: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # 10. 保存模型和tokenizer
    print(f"\n保存模型到 {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("✓ 训练完成!")


if __name__ == "__main__":
    main()
