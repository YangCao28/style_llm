import os
import shutil
from pathlib import Path

# Mapping of folder names to English author names
AUTHOR_MAPPING = {
    "倪匡": "Ni_Kuang",
    "古龙": "Gu_Long",
    "梁羽生": "Liang_Yusheng",
    "金庸": "Jin_Yong"
}

def get_author_english_name(path_part):
    for key, value in AUTHOR_MAPPING.items():
        if key in path_part:
            return value
    return None

def try_read_file(file_path):
    encodings = ['utf-8', 'gb18030', 'gbk', 'big5', 'utf-16']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read(), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None, None

def merge_lines(text):
    """
    Merges lines strictly based on user requirement:
    "Unless it is 。 at the end of a line, connect it."
    """
    if not text:
        return ""
        
    lines = text.splitlines()
    result_lines = []
    current_chunk = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        current_chunk.append(stripped)
        
        # Check if strictly ends with 。
        if stripped.endswith('。'):
            # Join current chunk and append as a line
            result_lines.append("".join(current_chunk))
            current_chunk = []
    
    # Process remaining text
    if current_chunk:
        result_lines.append("".join(current_chunk))
        
    return "\n".join(result_lines)

def main():
    source_dir = Path('data')
    target_base_dir = Path('new')
    
    # Recreate target directory
    if target_base_dir.exists():
        shutil.rmtree(target_base_dir)
    target_base_dir.mkdir()

    print(f"Starting processing from {source_dir} to {target_base_dir}...")
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith('.txt'):
                continue
                
            file_path = Path(root) / file
            
            # Determine author from the first folder under data/
            try:
                relative_path = file_path.relative_to(source_dir)
                author = relative_path.parts[0]
            except ValueError:
                print(f"Skipping {file_path}: Not under {source_dir}")
                continue
            
            # Construct new filename: {Author}_{OriginalFilename}
            # We treat the folder name in data/ as the author name (e.g. Gu_Long)
            original_filename = file_path.name
            base_filename = f"{author}_{original_filename}"
            target_path = target_base_dir / base_filename
            
            # Handle duplicates
            counter = 1
            while target_path.exists():
                stem = Path(base_filename).stem
                suffix = Path(base_filename).suffix
                target_path = target_base_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Read, Process, Write
            content, encoding = try_read_file(file_path)
            
            if content is None:
                print(f"Failed to decode {file_path}")
                continue

            # Fix newlines
            new_content = merge_lines(content)
                
            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                # print(f"Processed: {file_path.name} -> {target_path.name}") 
            except Exception as e:
                print(f"Error writing {target_path}: {e}")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
