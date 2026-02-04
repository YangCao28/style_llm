"""测试模型在没有指令的情况下是否还能执行任务

如果模型能在无指令下完成风格转换，说明它只是在做模式匹配，而非真正的指令遵循。
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen3-8B-Base", help="基座模型路径")
    parser.add_argument("--lora_model", required=True, help="LoRA模型路径")
    parser.add_argument("--test_data", default="data/modern_pairs_5000.jsonl", help="测试数据路径")
    parser.add_argument("--sample_line", type=int, default=10, help="测试样本的行号（从0开始）")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    args = parser.parse_args()

    print(f"🔧 配置:")
    print(f"  Base Model: {args.base_model}")
    print(f"  LoRA Model: {args.lora_model}")
    print(f"  Sample Line: {args.sample_line}")
    
    # 加载模型
    print(f"\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=torch.bfloat16)
    model.eval()
    
    # 加载测试数据
    print(f"\n📊 Loading test data...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    # 读取指定行号的样本
    if args.sample_line >= len(samples):
        print(f"❌ Line {args.sample_line} out of range! Total lines: {len(samples)}")
        return
    
    target_sample = samples[args.sample_line]
    
    print(f"✓ Loaded sample at line {args.sample_line}")
    print(f"  source_index: {target_sample.get('source_index', 'N/A')}")
    print(f"  record_id: {target_sample.get('record_id', 'N/A')}")
    user_content = target_sample["conversations"][1]["content"]
    modern_text_start = user_content.find("请将以下现代白话润色成雅致的文学文本：\n\n")
    if modern_text_start != -1:
        modern_text = user_content[modern_text_start + len("请将以下现代白话润色成雅致的文学文本：\n\n"):]
    else:
        modern_text = user_content
    
    expected_output = target_sample["conversations"][2]["content"]
    
    print(f"✓ Found sample {args.sample_id}")
    print(f"\n" + "="*80)
    print("实验设计:")
    print("="*80)
    print("如果模型在【无指令】情况下仍能输出文言风格，")
    print("说明它只学到了'看到白话→输出文言'的模式匹配，")
    print("而非真正理解指令。")
    print("="*80)
    
    # 测试1: 完全无指令，直接输入现代白话
    print(f"\n" + "="*80)
    print("测试1: 完全无指令 (裸文本输入)")
    print("="*80)
    prompt1 = modern_text
    print(f"\n输入 (前200字):")
    print(f"{prompt1[:200]}...")
    
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids1 = model.generate(
            **inputs1,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    output1 = tokenizer.decode(output_ids1[0][len(inputs1.input_ids[0]):], skip_special_tokens=True)
    print(f"\n输出 (前300字):")
    print(output1[:300])
    
    # 测试2: 只有system，无user instruction
    print(f"\n" + "="*80)
    print("测试2: 只有system消息 (无user指令)")
    print("="*80)
    system_msg = target_sample["conversations"][0]["content"]
    prompt2 = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{modern_text}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n输入格式:")
    print(f"<system>{system_msg[:100]}...</system>")
    print(f"<user>{modern_text[:100]}... (无'请润色'等指令)</user>")
    
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids2 = model.generate(
            **inputs2,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    output2 = tokenizer.decode(output_ids2[0][len(inputs2.input_ids[0]):], skip_special_tokens=True)
    print(f"\n输出 (前300字):")
    print(output2[:300])
    
    # 测试3: 标准指令（作为对照组）
    print(f"\n" + "="*80)
    print("测试3: 完整指令 (标准训练格式)")
    print("="*80)
    prompt3 = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n输入格式:")
    print(f"<system>{system_msg[:100]}...</system>")
    print(f"<user>请将以下现代白话润色成雅致的文学文本：\\n\\n{modern_text[:100]}...</user>")
    
    inputs3 = tokenizer(prompt3, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids3 = model.generate(
            **inputs3,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    output3 = tokenizer.decode(output_ids3[0][len(inputs3.input_ids[0]):], skip_special_tokens=True)
    print(f"\n输出 (前300字):")
    print(output3[:300])
    
    # 显示期望输出
    print(f"\n" + "="*80)
    print("期望输出 (训练数据中的原文)")
    print("="*80)
    print(expected_output[:300])
    
    # 分析
    print(f"\n" + "="*80)
    print("结论分析")
    print("="*80)
    print("如果测试1和测试2的输出都是文言风格且类似测试3，")
    print("说明模型【没有】真正学会指令遵循，只是在做:")
    print("  输入模式: 现代白话文本")
    print("  输出模式: 文言风格文本")
    print("\n如果测试1和测试2输出混乱或不像文言，而测试3正常，")
    print("说明模型【确实】在依赖指令来决定任务类型。")
    print("="*80)


if __name__ == "__main__":
    main()
