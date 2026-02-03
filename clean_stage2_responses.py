"""清理 Stage2 训练数据中 assistant 回复的"尾巴"

问题：训练数据中可能包含以下无用后缀：
- "改写完成"
- "请参考"
- "以上是改写结果"
- 等等

这些"尾巴"会让模型学到"说话不落句"的坏习惯。

Usage:
    python clean_stage2_responses.py \
        --input_path data/stage2_sample_5000.jsonl \
        --output_path data/stage2_sample_5000_cleaned.jsonl
"""

import argparse
import json
import re
from pathlib import Path


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
    original = response
    cleaned = response.strip()
    
    # 应用所有清理模式
    for pattern in TAIL_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # 再次去除尾部空白
    cleaned = cleaned.strip()
    
    # 如果清理后变化了，记录一下
    if cleaned != original:
        return cleaned, True
    return cleaned, False


def clean_dataset(input_path: Path, output_path: Path):
    """清理整个数据集"""
    print(f"📖 读取: {input_path}")
    
    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"✓ 加载了 {len(samples):,} 个样本")
    
    # 清理每个样本
    cleaned_count = 0
    for sample in samples:
        conversations = sample.get("conversations", [])
        for msg in conversations:
            role = msg.get("role") or msg.get("from")
            if role in ("assistant", "gpt", "bot"):
                content = msg.get("content") or msg.get("value") or ""
                cleaned_content, changed = clean_assistant_response(content)
                
                if changed:
                    cleaned_count += 1
                    # 更新内容
                    if "content" in msg:
                        msg["content"] = cleaned_content
                    if "value" in msg:
                        msg["value"] = cleaned_content
    
    print(f"✓ 清理了 {cleaned_count} 个 assistant 回复")
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✓ 保存到: {output_path}")
    
    # 显示统计
    print(f"\n📊 统计:")
    print(f"  总样本数: {len(samples):,}")
    print(f"  清理样本数: {cleaned_count:,}")
    print(f"  清理比例: {cleaned_count / len(samples) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="清理 Stage2 训练数据中的尾巴")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="输入的 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="输出的清理后 JSONL 文件路径",
    )
    
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"❌ 输入文件不存在: {args.input_path}")
        return
    
    clean_dataset(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
