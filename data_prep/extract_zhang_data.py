"""从combined_dataset中提取张恨水数据并清理章回体标记"""

import json
import re
from pathlib import Path

def is_zhang_henshui(record):
    """判断是否为张恨水数据"""
    source = record.get("source_file", "") + record.get("dataset", "")
    return "zhang" in source.lower() or "恨水" in source

def clean_chapter_marks(text):
    """清理章回体标记"""
    # 常见的章回体模式
    patterns = [
        r'第[一二三四五六七八九十百千零0-9]+章[^。\n]{0,30}[\n\s]*',  # 第X章 标题
        r'第[一二三四五六七八九十百千零0-9]+回[^。\n]{0,30}[\n\s]*',  # 第X回 标题
        r'^第[0-9]+章\s+[^\n]+\n',  # 阿拉伯数字章节
        r'第[0-9]+回\s+[^\n]+\n',
        r'【第[^】]+】',  # 【第X章】
        r'（第[^）]+）',  # （第X回）
        r'话说[当且]?[时日]',  # 话说当时、话说
        r'却说',
        r'且说',
        r'再说',
        r'书接上回',
        r'欲知后事如何',
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
    
    # 清理多余的空行
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
    
    return cleaned.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dataset/combined_dataset_uniform.jsonl")
    parser.add_argument("--output", type=str, default="data/dataset/zhang_cleaned.jsonl")
    parser.add_argument("--min_length", type=int, default=500, help="最小文本长度")
    parser.add_argument("--max_length", type=int, default=1500, help="最大文本长度")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"读取数据: {input_file}")
    
    zhang_records = []
    total_count = 0
    zhang_count = 0
    cleaned_count = 0
    filtered_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                record = json.loads(line)
                
                # 只要张恨水数据
                if not is_zhang_henshui(record):
                    continue
                
                zhang_count += 1
                text = record.get("text", "")
                
                # 清理章回体标记
                cleaned_text = clean_chapter_marks(text)
                
                # 检查是否有清理
                if cleaned_text != text:
                    cleaned_count += 1
                
                # 长度过滤
                if len(cleaned_text) < args.min_length or len(cleaned_text) > args.max_length:
                    filtered_count += 1
                    continue
                
                # 保存清理后的文本
                record['text'] = cleaned_text
                record['original_length'] = len(text)
                record['cleaned_length'] = len(cleaned_text)
                
                zhang_records.append(record)
                
            except Exception as e:
                print(f"处理记录时出错: {e}")
                continue
    
    print(f"\n统计:")
    print(f"  总记录数: {total_count}")
    print(f"  张恨水数据: {zhang_count}")
    print(f"  清理章回体: {cleaned_count}")
    print(f"  长度过滤掉: {filtered_count}")
    print(f"  最终保留: {len(zhang_records)}")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in zhang_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 输出文件: {output_file}")
    
    # 显示前3个样本
    print(f"\n前3个样本:")
    for i, record in enumerate(zhang_records[:3]):
        print(f"\n样本{i+1}:")
        print(f"  原始长度: {record['original_length']} 字")
        print(f"  清理后: {record['cleaned_length']} 字")
        print(f"  文本前100字: {record['text'][:100]}...")

if __name__ == "__main__":
    main()
