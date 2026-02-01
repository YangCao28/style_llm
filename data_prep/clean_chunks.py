import json
import re
from pathlib import Path

CLOSING_QUOTES = (
    "」", "』", "”", "’", "》", "】", "〉", '"', "'"
)

def clean_text(text):
    # Terms explicitly requested by the user to be removed
    # "首页", "字母", "下一页", "上一页", "网站"
    # Added some common variations often found with these
    garbage_terms = [
        "首页", 
        "字母", 
        "下一页", 
        "上一页", 
        "网站",
        "加入书架",
        "投推荐票",
        "回目录",
        "txt下载",
        "TXT下载",
        "本小章还未完",
        "请点击下一页继续阅读后面精彩内容",
        "请点击继续阅读后面精彩内容"
    ]
    
    cleaned = text
    for term in garbage_terms:
        # Case insensitive replacement for english terms if any, though most are Chinese
        cleaned = cleaned.replace(term, "")
        
    # Remove all English letters as requested
    cleaned = re.sub(r'[a-zA-Z]', '', cleaned)
        
    return cleaned

def shift_leading_closing_quotes(text, previous_record):
    if not text:
        return text

    idx = 0
    length = len(text)

    # skip initial whitespace
    while idx < length and text[idx].isspace():
        idx += 1

    start = idx
    moved = False

    while start < length and text[start] in CLOSING_QUOTES:
        if previous_record is None:
            break
        prev_text = previous_record.get('text', '')
        previous_record['text'] = prev_text + text[start]
        start += 1
        # skip whitespace between quotes and next content
        while start < length and text[start].isspace():
            start += 1
        moved = True

    if moved:
        return text[start:]
    return text

def process_file(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Cleaning {input_path} -> {output_path}...")
    
    removed_count = 0
    processed_count = 0
    previous_record = None
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                original_text = record.get('text', '')
                
                cleaned_text = clean_text(original_text)
                cleaned_text = shift_leading_closing_quotes(cleaned_text, previous_record)

                # If cleaning removed everything or left very little meaningful text
                if len(cleaned_text.strip()) < 5: 
                    removed_count += 1
                    continue
                
                # Write previous record if exists
                if previous_record:
                    outfile.write(json.dumps(previous_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                
                record['text'] = cleaned_text
                previous_record = record
                
            except json.JSONDecodeError:
                continue
        
        # Write the last record
        if previous_record:
            outfile.write(json.dumps(previous_record, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"Done. Processed {processed_count} chunks. Removed {removed_count} empty/short chunks.")

def main():
    # Clean wuxia chunks
    process_file(
        Path('./data/wuxia_chunks_cleaned.jsonl'),
        Path('./data/wuxia_chunks_cleaned1.jsonl')
    )
    

if __name__ == "__main__":
    main()
