"""评估模型在对抗测试集上的表现

重点指标：
- ROUGE-L: 衡量原样返回的准确度（期望接近1.0）
- 改写率: 多少样本被错误改写了（期望接近0%）

用法：
    python -m lora.evaluate_adversarial --base_model Qwen3-8B-Base --lora_model single_stage_v2 --test_file data/split/test_adversarial_200.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge import Rouge


def extract_modern_text(user_content: str) -> str:
    """从user content中提取现代白话文本"""
    # 移除各种可能的指令前缀
    prefixes = [
        "请原样输出以下内容，不要做任何修改：\n\n",
        "直接返回下面的文字，保持原样：\n\n",
        "请一字不改地输出以下内容：\n\n",
        "不要修改，原样返回：\n\n",
        "请保持原文不变，直接输出：\n\n",
    ]
    
    for prefix in prefixes:
        if prefix in user_content:
            return user_content[len(prefix):]
    
    return user_content


def build_prompt(system: str, user: str) -> str:
    """构建Qwen Chat格式的prompt"""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen3-8B-Base", help="基座模型路径")
    parser.add_argument("--lora_model", required=True, help="LoRA模型路径")
    parser.add_argument("--test_file", required=True, help="对抗测试集文件")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--output", type=str, default=None, help="保存详细结果的文件（可选）")
    args = parser.parse_args()

    print(f"🔧 配置:")
    print(f"  Base Model: {args.base_model}")
    print(f"  LoRA Model: {args.lora_model}")
    print(f"  Test File: {args.test_file}")
    
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
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(test_data)} samples")
    
    # 评估
    print(f"\n🧪 Evaluating...")
    rouge = Rouge()
    results = []
    rouge_scores = []
    rewrite_count = 0
    
    for i, sample in enumerate(test_data):
        system = sample["conversations"][0]["content"]
        user = sample["conversations"][1]["content"]
        expected = sample["conversations"][2]["content"]  # 应该是原样返回的现代白话
        
        # 构建prompt
        prompt = build_prompt(system, user)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 使用贪心解码，更稳定
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 解码
        generated = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # 清理
        if "<|im_end|>" in generated:
            generated = generated[:generated.find("<|im_end|>")]
        generated = generated.strip()
        
        # 计算ROUGE-L
        try:
            score = rouge.get_scores(generated, expected)[0]["rouge-l"]["f"]
            rouge_scores.append(score)
        except:
            score = 0.0
            rouge_scores.append(0.0)
        
        # 判断是否被错误改写（ROUGE-L < 0.8 认为是改写了）
        if score < 0.8:
            rewrite_count += 1
        
        results.append({
            "index": i,
            "record_id": sample.get("record_id"),
            "expected": expected[:100],
            "generated": generated[:100],
            "rouge_l": score,
            "is_rewritten": score < 0.8
        })
        
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(test_data)}")
    
    # 统计
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    rewrite_rate = rewrite_count / len(test_data) * 100
    
    print(f"\n" + "="*80)
    print("📊 评估结果")
    print("="*80)
    print(f"\n总样本数: {len(test_data)}")
    print(f"平均 ROUGE-L: {avg_rouge:.4f} (期望接近1.0)")
    print(f"错误改写数: {rewrite_count} / {len(test_data)}")
    print(f"错误改写率: {rewrite_rate:.2f}% (期望接近0%)")
    
    print(f"\n✅ 评估标准:")
    if avg_rouge >= 0.95:
        print(f"  ROUGE-L ≥ 0.95: 🎉 优秀！模型完美理解了对抗指令")
    elif avg_rouge >= 0.85:
        print(f"  ROUGE-L ≥ 0.85: ✅ 良好，模型基本掌握了指令遵循")
    elif avg_rouge >= 0.70:
        print(f"  ROUGE-L ≥ 0.70: ⚠️  一般，模型部分理解指令")
    else:
        print(f"  ROUGE-L < 0.70: ❌ 失败，模型仍在忽略指令")
    
    if rewrite_rate <= 5:
        print(f"  改写率 ≤ 5%: 🎉 优秀！几乎不误改")
    elif rewrite_rate <= 15:
        print(f"  改写率 ≤ 15%: ✅ 可接受")
    else:
        print(f"  改写率 > 15%: ⚠️  需要改进")
    
    # 显示几个典型案例
    print(f"\n📝 典型案例:")
    print("="*80)
    
    # 找出最好和最差的3个
    sorted_results = sorted(results, key=lambda x: x["rouge_l"], reverse=True)
    
    print(f"\n✅ 最好的3个案例:")
    for i, r in enumerate(sorted_results[:3]):
        print(f"\n案例 {i+1} (ROUGE-L: {r['rouge_l']:.4f})")
        print(f"  期望: {r['expected']}")
        print(f"  实际: {r['generated']}")
    
    print(f"\n❌ 最差的3个案例:")
    for i, r in enumerate(sorted_results[-3:]):
        print(f"\n案例 {i+1} (ROUGE-L: {r['rouge_l']:.4f})")
        print(f"  期望: {r['expected']}")
        print(f"  实际: {r['generated']}")
    
    # 保存详细结果
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_samples": len(test_data),
                    "avg_rouge_l": avg_rouge,
                    "rewrite_count": rewrite_count,
                    "rewrite_rate": rewrite_rate
                },
                "details": results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细结果已保存到: {output_file}")
    
    print("="*80)


if __name__ == "__main__":
    main()
