#!/usr/bin/env python3
"""
合并分块的JSONL文件

用法:
    python -m data_prep.merge_chunks --input data/modern_pairs_chunks --output data/modern_pairs_final.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="合并分块JSONL文件")
    parser.add_argument("--input", required=True, help="分块文件所在目录")
    parser.add_argument("--output", required=True, help="合并后的输出文件")
    parser.add_argument("--count", type=int, default=None, help="只合并前N条记录（None=全部）")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    # 获取所有分块文件并按编号排序
    chunk_files = sorted(input_dir.glob("chunk_*.jsonl"))
    
    if not chunk_files:
        print(f"❌ 在 {input_dir} 中未找到分块文件 (chunk_*.jsonl)")
        return

    print(f"📂 找到 {len(chunk_files)} 个分块文件")
    if args.count:
        print(f"🎯 限制合并前 {args.count} 条记录")
    
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for chunk_file in chunk_files:
            if args.count and total_count >= args.count:
                print(f"  已达到目标数量 {args.count}，停止合并")
                break
                
            print(f"  处理: {chunk_file.name}...", end=" ")
            chunk_count = 0
            
            with open(chunk_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if args.count and total_count >= args.count:
                        break
                    out_f.write(line)
                    chunk_count += 1
                    total_count += 1
            
            print(f"✓ {chunk_count} 条")
    
    print(f"\n✅ 合并完成！")
    print(f"   总计: {total_count} 条记录")
    print(f"   输出: {output_file}")


if __name__ == "__main__":
    main()
