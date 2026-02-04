"""清洗张恨水原文txt，生成干净的jsonl分块数据"""

import json
import re
from pathlib import Path
from typing import List

def is_chapter_header(line: str) -> bool:
    """判断是否为章回标题"""
    line = line.strip()
    if not line:
        return False
    
    # 匹配：第+数字（中文或阿拉伯）+回/章
    patterns = [
        r'^第[0-9零一二三四五六七八九十百千]+[回章]',
        r'^第\d+[回章]',
    ]
    
    for pattern in patterns:
        if re.match(pattern, line):
            return True
    return False


def should_remove(line: str) -> bool:
    """判断是否应该删除这一行"""
    line = line.strip()
    if not line:
        return False
    
    # 要删除的模式
    remove_patterns = [
        '序言', '前言', '后记', '自序', '序',
        '下一页', '上一页', '下一章', '上一章', '返回目录', 
        '回到首页', '返回总目录', '返回',
        '本章完', '未完待续', '待续',
        '下回分解', '欲知后事', '且听下回分解',
        '第一卷', '第二卷', '第三卷', '第四卷',
    ]
    
    for pattern in remove_patterns:
        if pattern in line:
            return True
    
    return False


def clean_line(line: str) -> str:
    """清理单行文本"""
    # 去除首尾空白
    line = line.strip()
    
    # 删除特定的叙事开头词
    line = re.sub(r'^[话却且再]说\s*', '', line)
    
    return line


def split_into_chunks(text: str, target_size: int = 800) -> List[str]:
    """
    将文本分割成约target_size字的chunk，确保完整句子结尾
    
    Args:
        text: 要分割的文本
        target_size: 目标chunk大小（字数），可以少一点
    
    Returns:
        chunk列表
    """
    chunks = []
    current_chunk = ""
    
    # 句子结束标记
    sentence_endings = {"。", "！", "？", "；", "…"}
    closing_quotes = set('」』》”’）】"\'')
    
    i = 0
    while i < len(text):
        char = text[i]
        current_chunk += char
        
        # 如果达到目标长度，且当前字符是句子结束符，就切分
        if len(current_chunk) >= target_size and char in sentence_endings:
            # 将紧随其后的收尾引号也包含进当前chunk
            lookahead = i + 1
            while lookahead < len(text) and text[lookahead] in closing_quotes:
                current_chunk += text[lookahead]
                lookahead += 1
                i += 1

            # 但至少要有一定长度（避免太短）
            if len(current_chunk) >= target_size * 0.7:  # 至少70%的目标长度
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        i += 1
    
    # 添加剩余的文本
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def process_txt_file(input_path: str, output_path: str, chunk_size: int = 800):
    """处理txt文件，生成jsonl"""
    
    print(f"读取文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    
    # 第一步：清理和过滤
    cleaned_lines = []
    removed_count = 0
    chapter_count = 0
    
    for line in lines:
        # 空行跳过
        if not line.strip():
            continue
        
        # 章回标题跳过
        if is_chapter_header(line):
            chapter_count += 1
            print(f"跳过章回标题: {line.strip()[:50]}")
            continue
        
        # 导航/无关内容跳过
        if should_remove(line):
            removed_count += 1
            continue
        
        # 清理这一行
        cleaned = clean_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)
    
    print(f"删除章回标题: {chapter_count}行")
    print(f"删除无关内容: {removed_count}行")
    print(f"保留有效行: {len(cleaned_lines)}行")
    
    # 第二步：合并成大文本
    full_text = ''.join(cleaned_lines)
    print(f"合并后总字数: {len(full_text)}")
    
    # 第三步：分割成512字左右的chunk
    print(f"\n开始分chunk（目标{chunk_size}字，完整句子结尾）...")
    chunks = split_into_chunks(full_text, target_size=chunk_size)
    
    print(f"生成chunk数量: {len(chunks)}")
    
    # 统计chunk长度分布
    lengths = [len(c) for c in chunks]
    print(f"Chunk长度统计:")
    print(f"  最小: {min(lengths)}字")
    print(f"  最大: {max(lengths)}字")
    print(f"  平均: {sum(lengths)//len(lengths)}字")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]}字")
    
    # 第四步：生成jsonl
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for i, chunk in enumerate(chunks):
        record = {
            "text": chunk,
            "source": "zhang_henshui",
            "length": len(chunk),
            "_id": f"zhang_clean_{i+1}"
        }
        records.append(record)
    
    # 写入jsonl
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！输出文件: {output_path}")
    print(f"   总记录数: {len(records)}")
    
    # 显示前3个样本
    print(f"\n{'='*80}")
    print("前3个样本预览:")
    print(f"{'='*80}")
    for i in range(min(3, len(records))):
        record = records[i]
        print(f"\n样本 {i+1} ({record['length']}字):")
        print(record['text'][:200] + "...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='清洗张恨水txt文件')
    parser.add_argument(
        '--input',
        type=str,
        default='data/text/张恨水小说全集（套装共37册）.txt',
        help='输入txt文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset/zhang_cleaned_new.jsonl',
        help='输出jsonl文件路径'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=800,
        help='分块的目标字数'
    )
    
    args = parser.parse_args()
    process_txt_file(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
