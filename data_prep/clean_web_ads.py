"""清理文本中的网站广告和推广信息

删除包含以下模式的行：
1. 同时包含 www 和 com 的行（大小写不敏感）
2. 包含"小说网"的行（大小写不敏感）
"""

import argparse
import json
import re
from pathlib import Path


def should_remove_line(line: str) -> bool:
    """判断是否应该删除该行"""
    # 转换全角字符为半角
    line_halfwidth = line.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    ))
    line_lower = line_halfwidth.lower()
    
    # 移除所有空格、特殊符号来检测变体广告（例如 大;学，生，小，说'网 → 大学生小说网）
    line_normalized = re.sub(r'[\s\.,;:：，。；\~～\-\—\_\t=、\$@`"\']+', '', line_halfwidth)
    line_normalized_lower = line_normalized.lower()
    
    # 检查是否同时包含 www 和 com（检测normalized版本，更可靠）
    if 'www' in line_normalized_lower and 'com' in line_normalized_lower:
        return True
    
    # 检查是否包含"小说网"或"大学生"（原始行或归一化行中都算）
    if (
        '小说网' in line
        or '大学生' in line
        or '小说网' in line_normalized
        or '大学生' in line_normalized
    ):
        return True
    
    # 检查是否是纯数字行（通常是广告ID）
    if line.strip() and line.strip().isdigit() and len(line.strip()) >= 4:
        return True
    
    # 检查变体广告模式（去除空格和符号后的关键字组合）
    # 使用更灵活的匹配：只要包含这些字符组合就删除
    ad_keywords = [
        ('小', '说', 't', 'x', 't'),  # 小说txt 的各种变体
        ('小', '说', '天', '堂'),      # 小说天堂
        ('t', 'x', 't', '小', '说'),   # txt小说
        ('更', '新', '最', '快'),
        ('手', '打'),
        ('顶', '点', '小', '说'),
        ('笔', '趣', '阁'),
        ('飘', '天', '文', '学'),
        ('大', '学', '生', '小', '说', '网'),  # 大学生小说网 及其被符号打散的变体
    ]
    
    for keywords in ad_keywords:
        # 检查所有关键字是否按顺序出现（允许中间有其他字符）
        pattern = '.*'.join(re.escape(k) for k in keywords)
        if re.search(pattern, line_normalized_lower):
            return True
    
    return False


def clean_chapter_marks(text: str) -> str:
    """清理章回体标记"""
    # 常见的章回体模式
    patterns = [
        r'第[一二三四五六七八九十百千零0-9]+章[^。\n]{0,30}[\n\s]*',  # 第X章 标题
        r'第[一二三四五六七八九十百千零0-9]+回[^。\n]{0,30}[\n\s]*',  # 第X回 标题
        r'^第[0-9]+章\s+[^\n]+\n',  # 阿拉伯数字章节
        r'第[0-9]+回\s+[^\n]+\n',
        r'【第[^】]+】',  # 【第X章】
        r'（第[^）]+）',  # （第X回）
        r'书接上回',
        r'欲知后事如何',
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
    
    # 清理多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def clean_text(text: str, remove_chapter_marks: bool = False) -> str:
    """清理文本，删除包含广告的行"""
    # 先处理HTML实体
    import html
    text = html.unescape(text)
    
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not should_remove_line(line)]
    result = '\n'.join(cleaned_lines)
    
    if remove_chapter_marks:
        result = clean_chapter_marks(result)
    
    return result


def clean_jsonl_file(input_path: Path, output_path: Path, text_field: str = 'text', remove_chapter_marks: bool = False):
    """清理 JSONL 文件中的文本字段"""
    print(f"读取: {input_path}")
    
    total_records = 0
    cleaned_records = 0
    removed_lines_count = 0
    
    with input_path.open('r', encoding='utf-8') as fin, \
         output_path.open('w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total_records += 1
            record = json.loads(line)
            
            # 获取文本字段
            if text_field not in record:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                continue
            
            original_text = record[text_field]
            cleaned_text = clean_text(original_text, remove_chapter_marks)
            
            # 统计删除的行数
            original_lines = len(original_text.split('\n'))
            cleaned_lines = len(cleaned_text.split('\n'))
            removed_lines_count += (original_lines - cleaned_lines)
            
            # 如果文本有变化，记录
            if cleaned_text != original_text:
                cleaned_records += 1
                record[text_field] = cleaned_text
            
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ 处理完成")
    print(f"  总记录数: {total_records:,}")
    print(f"  修改记录数: {cleaned_records:,}")
    print(f"  删除行数: {removed_lines_count:,}")
    if remove_chapter_marks:
        print(f"  已清理章回体标记")
    print(f"  输出: {output_path}")


def clean_txt_file(input_path: Path, output_path: Path, remove_chapter_marks: bool = False):
    """清理 TXT 文件中的广告行"""
    print(f"读取: {input_path}")
    
    with input_path.open('r', encoding='utf-8') as f:
        text = f.read()
    
    original_lines = text.split('\n')
    total_lines = len(original_lines)
    
    cleaned_text = clean_text(text, remove_chapter_marks)
    cleaned_lines = cleaned_text.split('\n')
    removed_lines_count = total_lines - len(cleaned_lines)
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"✓ 处理完成")
    print(f"  总行数: {total_lines:,}")
    print(f"  删除行数: {removed_lines_count:,}")
    print(f"  保留行数: {len(cleaned_lines):,}")
    if remove_chapter_marks:
        print(f"  已清理章回体标记")
    print(f"  输出: {output_path}")


def clean_json_file(input_path: Path, output_path: Path):
    """清理 JSON 对话文件中的 assistant 内容"""
    print(f"读取: {input_path}")
    
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_records = len(data)
    cleaned_records = 0
    removed_lines_count = 0
    
    for record in data:
        conversations = record.get('conversations', [])
        for msg in conversations:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'assistant' and content:
                original_lines = len(content.split('\n'))
                cleaned_content = clean_text(content)
                cleaned_lines = len(cleaned_content.split('\n'))
                
                if cleaned_content != content:
                    cleaned_records += 1
                    removed_lines_count += (original_lines - cleaned_lines)
                    msg['content'] = cleaned_content
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 处理完成")
    print(f"  总记录数: {total_records:,}")
    print(f"  修改记录数: {cleaned_records:,}")
    print(f"  删除行数: {removed_lines_count:,}")
    print(f"  输出: {output_path}")


def process_file(input_path: Path, output_path: Path, text_field: str, file_format: str, remove_chapter_marks: bool):
    """处理单个文件"""
    print(f"\n{'='*80}")
    print(f"处理: {input_path.name}")
    print(f"{'='*80}")
    
    if file_format == 'txt':
        clean_txt_file(input_path, output_path, remove_chapter_marks)
    elif file_format == 'jsonl':
        clean_jsonl_file(input_path, output_path, text_field, remove_chapter_marks)
    else:
        clean_json_file(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="清理文本中的网站广告信息")
    parser.add_argument('--input', type=Path, required=True, help="输入文件或文件夹路径")
    parser.add_argument('--output', type=Path, help="输出文件或文件夹路径（默认为输入路径_cleaned）")
    parser.add_argument('--text-field', default='text', help="JSONL文件中的文本字段名（默认: text）")
    parser.add_argument('--format', choices=['txt', 'jsonl', 'json'], default='txt', help="文件格式（默认: txt）")
    parser.add_argument('--pattern', default='*.txt', help="处理文件夹时的文件匹配模式（默认: *.txt）")
    parser.add_argument('--remove-chapter-marks', action='store_true', help="同时清理章回体标记（第X章、第X回等）")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ 路径不存在: {args.input}")
        return
    
    print(f"\n{'='*80}")
    print("清理网站广告和推广信息")
    print(f"{'='*80}")
    
    # 处理文件夹
    if args.input.is_dir():
        # 确定输出文件夹
        if args.output:
            output_dir = args.output
        else:
            output_dir = args.input.parent / f"{args.input.name}_cleaned"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输入文件夹: {args.input}")
        print(f"输出文件夹: {output_dir}")
        print(f"文件模式: {args.pattern}")
        
        # 查找所有匹配的文件
        files = list(args.input.glob(args.pattern))
        if not files:
            print(f"❌ 未找到匹配 {args.pattern} 的文件")
            return
        
        print(f"找到 {len(files)} 个文件\n")
        
        for input_file in files:
            output_file = output_dir / input_file.name
            try:
                process_file(input_file, output_file, args.text_field, args.format, args.remove_chapter_marks)
            except Exception as e:
                print(f"❌ 处理失败 {input_file.name}: {e}")
        
        print(f"\n✅ 全部完成！输出到: {output_dir}")
    
    # 处理单个文件
    else:
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = args.input.parent / f"{args.input.stem}_cleaned{args.input.suffix}"
        
        process_file(args.input, output_path, args.text_field, args.format, args.remove_chapter_marks)
    
    print()


if __name__ == '__main__':
    main()
