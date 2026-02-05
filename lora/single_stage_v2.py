"""单阶段指令微调 V2 - 支持Soft Masking

🔑 关键改进：
  1. 支持混合屏蔽法（20% Soft Masking）
  2. 可配置soft_mask_ratio
  3. 降低learning rate以配合soft masking

Usage:
    python -m lora.single_stage_v2 --config lora/single_stage_v2_config.json
"""

from __future__ import annotations

import argparse
import json
import random
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


def formatting_func_with_soft_mask(example, tokenizer, max_seq_length=2048, soft_mask_ratio=0.2, debug=False):
    """格式化对话数据 - 支持Soft Masking
    
    Args:
        soft_mask_ratio: Soft Masking比例，这部分样本不屏蔽user端（默认20%）
                        设为0.0则完全Hard Mask，设为1.0则全量学习
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
    
    # 分别tokenize
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    assistant_text = assistant_response + "<|im_end|>"
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    
    # 拼接完整序列
    input_ids = prompt_ids + assistant_ids
    
    # 截断
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        if len(prompt_ids) > max_seq_length:
            return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 🔥 混合屏蔽法：决定是否使用Soft Masking
    use_soft_mask = random.random() < soft_mask_ratio
    
    if use_soft_mask:
        # Soft Masking: 全量学习，不屏蔽user
        # 模型需要预测整段对话，有助于理解指令
        labels = input_ids.copy()
    else:
        # Hard Masking: 只计算assistant的loss
        labels = [-100] * len(prompt_ids) + assistant_ids
    
    labels = labels[:max_seq_length]
    
    # Padding
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * len(input_ids[:max_seq_length - padding_length]) + [0] * padding_length
    labels = labels + [-100] * (max_seq_length - len(labels))
    
    if debug:
        mask_type = "Soft (全量学习)" if use_soft_mask else "Hard (只学Assistant)"
        print(f"\n🔍 样本屏蔽类型: {mask_type}")
        print(f"  Prompt tokens: {len(prompt_ids)}")
        print(f"  Assistant tokens: {len(assistant_ids)}")
        print(f"  Labels中-100数量: {sum(1 for l in labels if l == -100)}")
        print(f"  Labels中有效数量: {sum(1 for l in labels if l != -100)}")
    
    return {
        "input_ids": input_ids[:max_seq_length],
        "attention_mask": attention_mask[:max_seq_length],
        "labels": labels[:max_seq_length],
    }


class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.training_losses.append(logs["loss"])
                self.steps.append(state.global_step)
                if getattr(state, "is_world_process_zero", True):
                    print(f"[step {state.global_step}] train_loss = {logs['loss']:.4f}")
            
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
                if getattr(state, "is_world_process_zero", True):
                    print(f"[step {state.global_step}] eval_loss = {logs['eval_loss']:.4f}")


class TestGenerationCallback(TrainerCallback):
    """每N步生成测试样本，供人工评估改写效果"""
    
    def __init__(self, model, tokenizer, test_prompts, test_interval=100, output_dir="."):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.test_interval = test_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def on_step_end(self, args, state, control, **kwargs):
        # 每test_interval步运行一次测试
        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self._run_test_generation(state.global_step)
    
    def _run_test_generation(self, step):
        """运行测试生成"""
        print(f"\n{'='*80}")
        print(f"🧪 [Step {step}] 运行测试生成 - 人工评估改写效果")
        print(f"{'='*80}")
        
        self.model.eval()
        test_results = []
        
        with torch.no_grad():
            for i, prompt in enumerate(self.test_prompts):
                print(f"\n--- 测试样本 {i+1}/{len(self.test_prompts)} ---")
                print(f"原文: {prompt[:100]}...")
                
                # 构建输入
                messages = [
                    {"role": "user", "content": f"请将下面的文言文改写为现代白话文：\n\n{prompt}"}
                ]
                input_text = "\n".join([
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
                    for msg in messages
                ]) + "\n<|im_start|>assistant\n"
                
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
                
                # 生成
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # 解码输出
                generated_text = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                print(f"改写: {generated_text[:200]}...")
                
                test_results.append({
                    "step": step,
                    "sample_id": i + 1,
                    "original": prompt,
                    "rewritten": generated_text
                })
        
        # 保存测试结果到JSON文件
        test_file = self.output_dir / f"test_generation_step_{step}.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 测试结果已保存到: {test_file}")
        print(f"{'='*80}\n")
        
        self.model.train()


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file")
    parser.add_argument("--soft_mask_ratio", type=float, default=None, help="Override soft masking ratio")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 参数覆盖
    if args.soft_mask_ratio is not None:
        config["soft_mask_ratio"] = args.soft_mask_ratio
    
    base_model_name = config["base_model_name"]
    dataset_path = config["dataset_path"]
    validation_dataset_path = config.get("validation_dataset_path")  # 可选验证集
    output_dir = config["output_dir"]
    soft_mask_ratio = config.get("soft_mask_ratio", 0.2)  # 默认20%
    
    print(f"\n🔧 配置:")
    print(f"  Base Model: {base_model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Validation: {validation_dataset_path or 'None'}")
    print(f"  Output: {output_dir}")
    print(f"  Soft Mask Ratio: {soft_mask_ratio:.1%} ({'混合屏蔽' if 0 < soft_mask_ratio < 1 else '全量学习' if soft_mask_ratio == 1 else 'Hard Mask'})")
    print(f"  Learning Rate: {config.get('learning_rate', 4e-5)}")
    print(f"  LoRA Rank: {config.get('lora_r', 64)}")
    
    # 加载tokenizer
    print(f"\n📦 加载 tokenizer...")
    tokenizer_path = config.get("tokenizer_path", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载base model
    print(f"📦 加载 base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print(f"\n📊 加载数据集: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✓ 加载 {len(dataset)} 条样本")
    
    # 格式化数据集 - 传入soft_mask_ratio
    print(f"\n🔄 格式化数据集 (Soft Mask Ratio={soft_mask_ratio:.1%})...")
    
    def format_fn(example):
        return formatting_func_with_soft_mask(
            example,
            tokenizer,
            max_seq_length=config.get("max_seq_length", 2048),
            soft_mask_ratio=soft_mask_ratio,
            debug=False
        )
    
    formatted_dataset = dataset.map(
        format_fn,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Formatting with Soft Masking"
    )
    
    # 过滤空样本
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"✓ 格式化完成: {len(formatted_dataset)} 条有效样本")
    
    # 加载验证集（如果提供）
    formatted_eval_dataset = None
    if validation_dataset_path:
        print(f"\n📊 加载验证集: {validation_dataset_path}")
        eval_dataset = load_dataset("json", data_files=validation_dataset_path, split="train")
        print(f"✓ 加载 {len(eval_dataset)} 条验证样本")
        
        formatted_eval_dataset = eval_dataset.map(
            format_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=1,
            desc="Formatting validation set"
        )
        formatted_eval_dataset = formatted_eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        print(f"✓ 验证集格式化完成: {len(formatted_eval_dataset)} 条有效样本")
    
    # 训练参数
    eval_steps = config.get("eval_steps", 100)  # 默认每100步评估
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 2.0),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 4e-5),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        logging_steps=config.get("logging_steps", 10),
        eval_strategy="steps" if formatted_eval_dataset else "no",
        eval_steps=eval_steps if formatted_eval_dataset else None,
        save_strategy="steps",
        save_steps=config.get("save_steps", 100),  # 每100步保存checkpoint
        save_total_limit=config.get("save_total_limit", 5),  # 保留最近5个checkpoint
        load_best_model_at_end=True if formatted_eval_dataset else False,
        metric_for_best_model="eval_loss" if formatted_eval_dataset else None,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
    )
    
    # 准备测试样本（用于人工评估）
    test_prompts = config.get("test_prompts", [
        "话说天下大势，分久必合，合久必分。",
        "却说玄德引军前进，忽报前面有一军阻路。",
        "且说曹操引兵至赤壁，与周瑜相拒。",
    ])
    
    # 回调
    loss_recorder = LossRecorderCallback()
    test_callback = TestGenerationCallback(
        model=model,
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        test_interval=config.get("test_interval", 100),  # 每100步测试
        output_dir=output_dir
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        eval_dataset=formatted_eval_dataset,
        callbacks=[loss_recorder, test_callback],
    )
    
    # 训练
    resume_checkpoint = args.resume_from_checkpoint or config.get("resume_from_checkpoint")
    print(f"\n🚀 开始训练...")
    if resume_checkpoint:
        print(f"  📂 从checkpoint恢复: {resume_checkpoint}")
    print(f"  {'='*80}")
    print(f"  🎯 关键配置:")
    print(f"     - Soft Masking: {soft_mask_ratio:.1%} 样本全量学习")
    print(f"     - Hard Masking: {(1-soft_mask_ratio):.1%} 样本只学Assistant")
    print(f"     - Learning Rate: {config.get('learning_rate', 4e-5)} (温和以配合Soft Masking)")
    print(f"  {'='*80}\n")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 保存
    print(f"\n💾 保存模型...")
    trainer.save_model(output_dir)
    
    # 显示训练曲线
    if loss_recorder.training_losses:
        print(f"\n📊 训练统计:")
        print(f"  初始 train_loss: {loss_recorder.training_losses[0]:.4f}")
        print(f"  最终 train_loss: {loss_recorder.training_losses[-1]:.4f}")
        print(f"  Train Loss 下降: {loss_recorder.training_losses[0] - loss_recorder.training_losses[-1]:.4f}")
        
        if loss_recorder.eval_losses:
            print(f"  初始 eval_loss: {loss_recorder.eval_losses[0]:.4f}")
            print(f"  最终 eval_loss: {loss_recorder.eval_losses[-1]:.4f}")
            print(f"  Eval Loss 下降: {loss_recorder.eval_losses[0] - loss_recorder.eval_losses[-1]:.4f}")
    
    # 保存loss曲线到JSON文件
    loss_history = {
        "train": {
            "steps": loss_recorder.steps,
            "losses": loss_recorder.training_losses
        },
        "eval": {
            "steps": loss_recorder.eval_steps,
            "losses": loss_recorder.eval_losses
        }
    }
    
    loss_file = Path(output_dir) / "loss_history.json"
    with open(loss_file, 'w', encoding='utf-8') as f:
        json.dump(loss_history, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 训练完成！")
    print(f"  📁 模型保存到: {output_dir}")
    print(f"  📊 Loss曲线保存到: {loss_file}")


if __name__ == "__main__":
    main()
