"""添加负样本（总结任务）到训练数据中

用法:
    python -m data_prep.add_negative_samples --input data/modern_pairs_5000.jsonl --output data/mixed_train.jsonl --negative_count 200
"""

import argparse
import json
import random
from pathlib import Path


SUMMARY_SYSTEM = "你是一个专业的文本摘要助手，擅长提取核心要点。"

SUMMARY_USER_TEMPLATE = "请用一句话总结以下内容的核心要点：\n\n{text}"


def create_summary(original_text: str) -> str:
    """根据原文创建简短摘要（规则生成，避免调用API）"""
    # 简单策略：提取前50字 + "等内容" 
    # 实际使用时可以调用LLM生成更好的摘要
    text = original_text.strip()
    
    # 尝试找到第一个句子
    for delimiter in ['。', '！', '？']:
        if delimiter in text[:100]:
            first_sentence = text[:text.find(delimiter) + 1]
            if len(first_sentence) > 20:
                return f"本段主要讲述了{first_sentence[:50]}的相关内容。"
    
    # 如果没找到句号，就取前30字
    return f"本段主要讲述了{text[:30]}等内容。"


def main():
    parser = argparse.ArgumentParser(description="添加负样本到训练数据")
    parser.add_argument("--input", required=True, help="原始训练数据")
    parser.add_argument("--output", required=True, help="混合后的输出文件")
    parser.add_argument("--negative_count", type=int, default=200, help="负样本数量")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    print(f"📖 读取原始数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        positive_samples = [json.loads(line) for line in f]

    print(f"✓ 加载 {len(positive_samples)} 条正样本")

    # 随机选择N条样本创建负样本
    selected = random.sample(positive_samples, min(args.negative_count, len(positive_samples)))
    
    negative_samples = []
    for item in selected:
        # 从原始样本中提取文学原文
        original_text = item["conversations"][2]["content"]
        
        # 创建总结任务
        summary = create_summary(original_text)
        
        negative_sample = {
            "source_index": item["source_index"],
            "record_id": f"{item['record_id']}_summary",
            "conversations": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(text=original_text)},
                {"role": "assistant", "content": summary}
            ]
        }
        negative_samples.append(negative_sample)

    print(f"✓ 生成 {len(negative_samples)} 条负样本（总结任务）")

    # 合并并打乱
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    print(f"✓ 混合后总计: {len(all_samples)} 条")

    # 写入输出
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成！")
    print(f"   正样本: {len(positive_samples)}")
    print(f"   负样本: {len(negative_samples)}")
    print(f"   输出: {output_file}")


if __name__ == "__main__":
    main()
