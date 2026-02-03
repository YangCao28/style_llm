"""诊断脚本：检测模型是否还在"续写对话"

Usage:
    python -m lora.diagnose_continuation \
        --model_name_or_path stage2_instruction_tuning/checkpoint-158
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_test_prompt():
    """构建一个简单的测试 prompt"""
    return """<|im_start|>system
你是一个文学改写助手，只负责调整文风和措辞。改写时必须完整保留原文提供的全部信息，不得扩写情节，也不得删减内容。你的回复必须在完成改写后立即结束，不得继续生成任何对话、提示或新任务。<|im_end|>
<|im_start|>user
请在不增删任何信息的前提下，用更紧张、悬疑的文风改写下面这段：
今天天气很好，小明出门去公园散步。<|im_end|>
<|im_start|>assistant
"""


def diagnose(model_path: str, attn_impl: str = "sdpa"):
    """运行诊断测试"""
    print("=" * 80)
    print("🔍 Stage2 续写问题诊断")
    print("=" * 80)
    print(f"\n模型路径: {model_path}")
    
    # 加载模型和 tokenizer
    print("\n📦 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    
    # 构建测试 prompt
    prompt = build_test_prompt()
    print("\n📝 测试 Prompt:")
    print(prompt)
    print("\n" + "-" * 80)
    
    # 测试 1: 不使用 stop tokens（旧行为）
    print("\n🧪 测试 1: 不使用 stop tokens（旧行为）")
    print("-" * 80)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # 使用贪婪解码，结果更确定
            pad_token_id=tokenizer.eos_token_id,
        )
    
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_response = completion[len(prompt):]
    
    print("📤 模型输出:")
    print(assistant_response[:500])
    
    # 检测问题标记
    issues = []
    if "\n<|im_start|>user" in assistant_response.lower() or "\nuser\n" in assistant_response.lower():
        issues.append("❌ 检测到 'user' 标记 - 模型试图开启新对话")
    if "\n<|im_start|>system" in assistant_response.lower() or "\nsystem\n" in assistant_response.lower():
        issues.append("❌ 检测到 'system' 标记 - 模型试图开启新对话")
    if "\n<|im_start|>assistant" in assistant_response.lower() or "\nassistant\n" in assistant_response.lower():
        issues.append("❌ 检测到多个 'assistant' 标记 - 模型试图继续对话")
    
    # 检查是否有 <|im_end|> 后仍继续生成
    if "<|im_end|>" in assistant_response:
        end_pos = assistant_response.find("<|im_end|>")
        after_end = assistant_response[end_pos + len("<|im_end|>"):].strip()
        if len(after_end) > 10:  # 超过 10 个字符认为是继续生成
            issues.append(f"❌ <|im_end|> 后仍生成了 {len(after_end)} 个字符")
    
    if not issues:
        print("\n✅ 测试 1 通过：模型没有尝试继续对话")
    else:
        print("\n⚠️  测试 1 发现问题:")
        for issue in issues:
            print(f"  {issue}")
    
    # 测试 2: 使用 stop tokens（新行为）
    print("\n" + "=" * 80)
    print("🧪 测试 2: 使用 stop tokens（修复后）")
    print("-" * 80)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )
    
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_response = completion[len(prompt):]
    
    print("📤 模型输出:")
    print(assistant_response[:500])
    
    # 检查是否正确停止
    if "<|im_end|>" in assistant_response:
        end_pos = assistant_response.find("<|im_end|>")
        after_end = assistant_response[end_pos + len("<|im_end|>"):].strip()
        if len(after_end) == 0:
            print("\n✅ 测试 2 通过：模型在 <|im_end|> 后正确停止")
        else:
            print(f"\n⚠️  <|im_end|> 后仍有 {len(after_end)} 个字符（但已大幅改善）")
    else:
        print("\n⚠️  未检测到 <|im_end|> - 可能生成被截断")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 诊断总结")
    print("=" * 80)
    
    if not issues:
        print("\n🎉 恭喜！模型行为正常，无需重新训练。")
        print("   - 建议：使用修复后的测试脚本 (test_stage2_instruction.py)")
    else:
        print("\n⚠️  检测到行为边界问题，建议采取以下措施：")
        print("\n   立即改善（无需重训）:")
        print("   1. 使用 test_stage2_instruction.py（已包含 stop tokens）")
        print("   2. 在 system prompt 中添加硬约束")
        print("\n   彻底解决（推荐）:")
        print("   1. 准备 20% 数据子集: python prepare_correction_subset.py")
        print("   2. 运行修正训练: python -m lora.stage2_instruction_tuning_fixed \\")
        print("                       --config lora/stage2_correction_config.json")
        print("\n   详见: lora/STAGE2_FIX_README.md")


def main():
    parser = argparse.ArgumentParser(description="诊断 Stage2 模型续写问题")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Stage2 模型路径",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        help="Attention implementation",
    )
    
    args = parser.parse_args()
    diagnose(args.model_name_or_path, args.attn_impl)


if __name__ == "__main__":
    main()
