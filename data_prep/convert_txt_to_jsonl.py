"""将清理后的txt文件转换成jsonl分块数据"""

import json
import re
from pathlib import Path
from typing import List
import argparse


def split_into_chunks(text: str, target_size: int = 800) -> List[str]:
    """
    将文本分割成约target_size字的chunk，确保完整句子结尾
    
    Args:
        text: 要分割的文本
        target_size: 目标chunk大小（字数）
    
    Returns:
        chunk列表
    """
    chunks = []
    current_chunk = ""
    
    # 句子结束标记（中英文常见终止符）
    sentence_endings = {"。", "！", "？", "；", "…", ".", "!", "?"}
    # 句末可能跟随的收尾引号/括号，尽量包含常见中英文符号
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


def extract_book_info(filename: str) -> dict:
    """从文件名提取书名和作者信息

    主要适配形如：
    - 书名_作者_TXT小说天堂.txt
    - 书名___作者-TXT小说天堂.txt
    """
    base = Path(filename).stem  # 去掉扩展名

    # 用正则更稳健地提取书名和作者
    m = re.match(r"^(?P<book>.+?)[_]+(?P<author>[^_-]+)(?:[-_]+TXT小说天堂)?$", base)
    if m:
        book_name = m.group("book")
        author = m.group("author")
    else:
        book_name = base
        author = "unknown"

    return {"book_name": book_name, "author": author}


def process_txt_file(input_path: Path, chunk_size: int = 800) -> List[dict]:
    """处理单个txt文件，返回该书的所有chunk记录，不直接写文件"""
    
    print(f"\n处理文件: {input_path.name}")
    
    # 读取文本
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    if not text:
        print(f"  ⚠️  文件为空，跳过")
        return []
    
    print(f"  原始字数: {len(text)}")
    
    # 提取书名和作者
    book_info = extract_book_info(input_path.name)
    
    # 分割成chunk
    chunks = split_into_chunks(text, target_size=chunk_size)
    print(f"  生成chunk数: {len(chunks)}")

    if not chunks:
        print(f"  ⚠️  没有生成chunk，跳过")
        return []
    
    # 统计chunk长度
    lengths = [len(c) for c in chunks]
    print(f"  Chunk长度: 最小{min(lengths)} 最大{max(lengths)} 平均{sum(lengths)//len(lengths)}")

    # 生成记录列表，由外层按作者写入
    base_id = book_info["book_name"].replace("/", "_").replace("\\", "_")
    records: List[dict] = []
    for i, chunk in enumerate(chunks):
        record = {
            "text": chunk,
            "source": book_info["book_name"],
            "author": book_info["author"],
            "length": len(chunk),
            "_id": f"{base_id}_{i+1}",
        }
        records.append(record)

    print(f"  ✅ 完成，返回 {len(records)} 条记录")
    return records


def process_directory(input_dir: Path, output_dir: Path, chunk_size: int = 800, pattern: str = "*.txt"):
    """处理目录下所有txt文件，并按作者聚合到一个jsonl中"""
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有txt文件
    txt_files = sorted(input_dir.glob(pattern))
    
    if not txt_files:
        print(f"❌ 在 {input_dir} 中没有找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    print(f"{'='*80}")
    
    total_chunks = 0
    success_count = 0
    # author -> List[record]
    author_records: dict[str, List[dict]] = {}

    for txt_file in txt_files:
        try:
            records = process_txt_file(txt_file, chunk_size)
            if records:
                total_chunks += len(records)
                success_count += 1
                author = records[0].get("author", "unknown") or "unknown"
                author_records.setdefault(author, []).extend(records)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")

    # 按作者写出一个jsonl
    print(f"\n开始按作者写出jsonl...")
    for author, records in author_records.items():
        safe_author = author.replace("/", "_").replace("\\", "_") or "unknown"
        output_file = output_dir / f"{safe_author}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  作者 {author}: {len(records)} 条，输出文件: {output_file.name}")
    
    print(f"\n{'='*80}")
    print(f"✅ 完成！")
    print(f"   成功处理: {success_count}/{len(txt_files)} 个文件")
    print(f"   总chunk数: {total_chunks}")
    print(f"   作者数: {len(author_records)}")
    print(f"   输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='将txt文件批量转换为jsonl格式')
    parser.add_argument(
        '--input',
        type=str,
        default='data/dataset_cleaned',
        help='输入目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset_jsonl',
        help='输出目录路径'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=800,
        help='分块的目标字数（默认800）'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.txt',
        help='文件匹配模式（默认*.txt）'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    process_directory(input_dir, output_dir, args.chunk_size, args.pattern)


if __name__ == "__main__":
    main()
