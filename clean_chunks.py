import json
import re
from pathlib import Path

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
        "TXT下载"
    ]
    
    cleaned = text
    for term in garbage_terms:
        # Case insensitive replacement for english terms if any, though most are Chinese
        cleaned = cleaned.replace(term, "")
        
    # Remove all English letters as requested
    cleaned = re.sub(r'[a-zA-Z]', '', cleaned)
        
    return cleaned

def main():
    input_path = Path('data_prep/wuxia_chunks.jsonl')
    output_path = Path('data_prep/wuxia_chunks_cleaned.jsonl')
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Cleaning {input_path} -> {output_path}...")
    
    removed_count = 0
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                original_text = record.get('text', '')
                
                cleaned_text = clean_text(original_text)
                
                # If cleaning removed everything or left very little meaningful text
                if len(cleaned_text.strip()) < 5: 
                    removed_count += 1
                    continue
                
                record['text'] = cleaned_text
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError:
                continue

    print(f"Done. Processed {processed_count} chunks. Removed {removed_count} empty/short chunks.")

if __name__ == "__main__":
    main()
