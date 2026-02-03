"""准备 Stage2 correction 训练的数据子集 (10-20%)

这个脚本从完整的 Stage2 数据集中随机抽取 10-20% 的样本，
用于轻量级的"行为校正"训练。

Usage:
    python prepare_correction_subset.py \
        --input_path data/stage2_sample_5000.jsonl \
        --output_path data/stage2_sample_subset_1000.jsonl \
        --ratio 0.2
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="准备 Stage2 correction 数据子集")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/stage2_sample_5000.jsonl"),
        help="输入的完整 Stage2 数据集路径",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/stage2_sample_subset_1000.jsonl"),
        help="输出的子集路径",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="采样比例 (0.1-0.3 recommended for correction training)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"❌ 输入文件不存在: {args.input_path}")
        return
    
    random.seed(args.seed)
    
    # 读取所有样本
    print(f"📖 Reading from: {args.input_path}")
    samples = []
    with args.input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"✓ Loaded {len(samples):,} samples")
    
    # 随机采样
    num_samples = int(len(samples) * args.ratio)
    selected_samples = random.sample(samples, num_samples)
    
    print(f"✓ Selected {num_samples:,} samples ({args.ratio:.1%})")
    
    # 保存
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved to: {args.output_path}")
    print("\n📋 使用方法:")
    print(f"    python -m lora.stage2_instruction_tuning \\")
    print(f"        --config lora/stage2_correction_config.json")


if __name__ == "__main__":
    main()
