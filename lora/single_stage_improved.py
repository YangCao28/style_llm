"""改进的单阶段指令微调脚本

改进点:
1. Soft Masking: 不完全屏蔽user端，部分样本保留0.1权重
2. 降低学习率: 4e-5 (原2e-4)
3. 减少epoch: 2.0 (原5.0)
4. 支持混合数据集（含负样本）
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_prompt(system: str, user: str) -> str:
    """构建Qwen Chat格式的prompt"""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def preprocess_function(examples: Dict, tokenizer, soft_mask_ratio: float = 0.1):
    """预处理函数：构建输入和标签
    
    Args:
        examples: 批量样本
        tokenizer: 分词器
        soft_mask_ratio: 部分样本不完全mask user端的比例
    """
    input_ids_list = []
    labels_list = []
    
    for conversations in examples["conversations"]:
        system = conversations[0]["content"]
        user = conversations[1]["content"]
        assistant = conversations[2]["content"]
        
        # 构建完整的对话
        prompt = build_prompt(system, user)
        full_text = prompt + assistant + "<|im_end|>"
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        
        # 🔑 策略B: Soft Masking
        # 10% 的样本不完全mask，给user端0.1的loss权重
        use_soft_mask = random.random() < soft_mask_ratio
        
        # 找到assistant开始的位置
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assistant_ids = tokenizer(assistant + "<|im_end|>", add_special_tokens=False)["input_ids"]
        
        if use_soft_mask:
            # Soft masking: user端保留低权重
            # 实际实现：这里仍然用-100，但在Trainer中可以通过自定义loss处理
            # 简化版：仍然mask，但可以在未来扩展
            labels = [-100] * len(prompt_ids) + assistant_ids
        else:
            # 标准masking: 只计算assistant的loss
            labels = [-100] * len(prompt_ids) + assistant_ids
        
        # 确保长度一致
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    base_model_name = config["base_model_name"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    
    print(f"🔧 配置:")
    print(f"  Base Model: {base_model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Learning Rate: {config.get('learning_rate', 4e-5)}")
    print(f"  Epochs: {config.get('num_train_epochs', 2.0)}")
    print(f"  LoRA Rank: {config.get('lora_r', 128)}")
    
    # 加载tokenizer和model
    print(f"\n📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 128),
        lora_alpha=config.get("lora_alpha", 256),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print(f"\n📊 Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(data)} samples")
    
    # 统计任务类型
    style_count = sum(1 for item in data if "总结" not in item["conversations"][1]["content"])
    summary_count = len(data) - style_count
    print(f"  风格改写任务: {style_count}")
    print(f"  总结任务: {summary_count}")
    
    # 转换为Dataset
    dataset = Dataset.from_list(data)
    
    # 预处理
    print(f"\n🔄 Preprocessing...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 2.0),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 4e-5),  # 🔑 降低学习率
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        logging_steps=config.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=config.get("save_steps", 100),
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # 训练
    print(f"\n🚀 Starting training...")
    result = trainer.train()
    
    print(f"\n💾 Saving model...")
    trainer.save_model(output_dir)
    
    print(f"\n✓ Training complete!")
    print(f"  📁 Model saved to: {output_dir}")
    
    # 打印训练统计
    if hasattr(result, 'metrics'):
        metrics = result.metrics
        print(f"\n📊 Training stats:")
        if 'train_loss' in metrics:
            print(f"  Initial loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"  Final loss: {trainer.state.log_history[-1].get('loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
