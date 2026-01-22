import os
import re
import json
from pathlib import Path

def split_into_sentences(text):
    """
    Splits text into sentences based on common Chinese punctuation.
    Keeps the punctuation with the sentence.
    """
    # Pattern matches:
    # [^。！？…\n]+ : One or more non-sentence-ending chars
    # [。！？…\n]+ : One or more sentence-ending chars
    # We use capturing group to keep delimiters, but we need to handle the logic to attach them back.
    
    # Simpler approach: split by specific delimiters and keep them.
    # The pattern ([。！？…\n]+) splits and captures the delimiter.
    # \n is included because sometimes a line break is a hard stop.
    
    pattern = r'([^。！？…\n]+[。！？…\n]+)'
    chunks = re.split(pattern, text)
    
    # Filter out empty strings
    sentences = [c for c in chunks if c]
    
    # Sometimes there might be leftover text at the end without punctuation
    # re.split might leave empty strings or separate text/punct.
    # Let's try a more robust generator.
    return sentences

def accumulate_chunks(sentences, max_chars=800):
    """
    Groups sentences into chunks <= max_chars.
    Note: max_chars is set to 800 to be safely within 1024 tokens for most tokenizers.
    """
    current_chunk = []
    current_len = 0
    chunks = []
    
    for sent in sentences:
        sent_len = len(sent)
        
        # If a single sentence is too long, we have to cut it or accept it.
        # Here we accept it to avoid breaking coherence, unless it's huge?
        # Ideally, we add it to the current chunk if it fits.
        
        if current_len + sent_len > max_chars:
            if current_chunk:
                chunks.append("".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
            
    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return chunks

def main():
    source_dir = Path('new')
    target_dir = Path('data_prep')
    output_file = target_dir / 'wuxia_chunks.jsonl'
    
    print(f"Reading from {source_dir}, saving to {output_file}...")
    
    files = list(source_dir.glob('**/*.txt'))
    print(f"Found {len(files)} files.")
    
    total_chunks = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
            if not text:
                continue
                
            # 1. Split into sentences (simple logic fix for re.split behavior)
            # re.split with capture groups returns [text, delimiter, text, delimiter...]
            # Actually, my previous regex `([^...]+[...]+)` captures the whole sentence+punct unit.
            # Example: "Hi. Bye." -> ["Hi.", "Bye.", ""]
            
            sentences = split_into_sentences(text)
            
            # 2. Group into chunks
            file_chunks = accumulate_chunks(sentences, max_chars=800)
            
            # 3. Write to JSONL
            filename = file_path.name
            author = filename.split('_')[0] if '_' in filename else "Unknown" # Rough heuristic logic from filename
            
            for i, chunk in enumerate(file_chunks):
                record = {
                    "id": f"{filename}_{i}",
                    "author": author,
                    "source_file": filename,
                    "text": chunk
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            total_chunks += len(file_chunks)
            if total_chunks % 1000 == 0:
                print(f"Processed {total_chunks} chunks...", end='\r')
                
    print(f"\nCompleted. Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()
