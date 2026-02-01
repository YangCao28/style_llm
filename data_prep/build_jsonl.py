import json
import re
from pathlib import Path
from typing import Callable, List

class Tokenizer:
    """Lightweight tokenizer wrapper to count tokens and decode slices."""
    def __init__(self, encode: Callable[[str], List], decode: Callable[[List], str], name: str):
        self.encode = encode
        self.decode = decode
        self.name = name


def get_tokenizer() -> Tokenizer:
    """Use tiktoken if available; otherwise fall back to char-based counting."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return Tokenizer(enc.encode, enc.decode, "tiktoken:cl100k_base")
    except Exception:
        return Tokenizer(lambda s: list(s), lambda tokens: "".join(tokens), "char-length")


def read_with_fallback(path: Path) -> str:
    encodings = ["utf-8", "gb18030", "gbk", "big5", "utf-16"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeError:
            continue
    raise UnicodeError(f"Unable to decode {path} with tried encodings: {encodings}")


def clean_text(text: str) -> str:
    """Remove garbage terms and invalid content, normalize quotes."""
    garbage_terms = [
        "首页", "字母", "下一页", "上一页", "网站",
        "加入书架", "投推荐票", "回目录",
        "txt下载", "TXT下载", "下载",
        "本小章还未完", "请点击下一页继续阅读后面精彩内容",
        "请点击继续阅读后面精彩内容"
    ]
    
    cleaned = text
    for term in garbage_terms:
        cleaned = cleaned.replace(term, "")
    
    # Remove all English letters
    cleaned = re.sub(r'[a-zA-Z]', '', cleaned)
    
    # Convert English quotes to Chinese quotes
    cleaned = cleaned.replace('"', '"').replace('"', '"')
    cleaned = cleaned.replace("'", ''').replace("'", ''')
    
    return cleaned


def normalize_text(raw_text: str) -> str:
    """Drop empty lines and trim whitespace."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)


def split_sentences(text: str) -> List[str]:
    """Split text on sentence-ending punctuation."""
    # Split on all closing punctuation including Chinese quotes
    parts = re.split(r'([。！？!?」』""\'>》}】\]""''])', text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        body = parts[i].strip()
        if not body:
            continue
        delimiter = parts[i + 1] if i + 1 < len(parts) else ""
        sentences.append(body + delimiter)
    return sentences


def force_split(sentence: str, tokenizer: Tokenizer, max_tokens: int) -> List[str]:
    """Hard-split a long sentence by tokens; only used when absolutely necessary."""
    token_ids = tokenizer.encode(sentence)
    slices = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_tokens = token_ids[start:start + max_tokens]
        slices.append(tokenizer.decode(chunk_tokens))
    return slices


def chunk_sentences(sentences: List[str], tokenizer: Tokenizer, max_tokens: int = 512) -> List[str]:
    """Group sentences into chunks. Ensure chunks end with proper punctuation."""
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    
    # Closing punctuation markers - including all Chinese closing quotes
    closing_punct = {'。', '！', '？', '!', '?', '」', '』', '"', '"', ''', ''', '>', '》', '}', '】', ']', '"', '”', '’'}
    # Closing quotes that should attach to the previous chunk if possible
    closing_quotes = ['」', '』', '"', '"', ''', ''', '”', '’']

    def add_chunk(text: str) -> None:
        text = text.strip()
        if not text:
            return
        # If chunk starts with closing quote, append it to previous chunk
        while text and text[0] in closing_quotes and chunks:
            chunks[-1] += text[0]
            text = text[1:].lstrip()
        if text:
            chunks.append(text)

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent))

        # If a single sentence exceeds the limit, split it forcibly.
        if sent_tokens > max_tokens:
            if current:
                joined = "".join(current)
                add_chunk(joined)
                current = []
                current_tokens = 0
            force_parts = force_split(sent, tokenizer, max_tokens)
            for part in force_parts:
                add_chunk(part)
            continue

        # Would adding this sentence exceed max_tokens?
        if current and current_tokens + sent_tokens > max_tokens:
            # Check if current chunk ends with closing punctuation
            joined = "".join(current)
            last_char = joined[-1] if joined else ""
            
            if last_char in closing_punct:
                add_chunk(joined)
                current = [sent]
                current_tokens = sent_tokens
            else:
                # Add one more sentence to complete the thought
                current.append(sent)
                joined = "".join(current)
                add_chunk(joined)
                current = []
                current_tokens = 0
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        joined = "".join(current)
        add_chunk(joined)

    return chunks


def write_jsonl(chunks: List[str], output_path: Path, source_name: str, prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 10:  # Skip very short chunks
                continue
            record = {
                "id": f"{prefix}_{idx:05d}",
                "source_file": source_name,
                "text": chunk,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_file(source_path: Path, output_dir: Path, prefix: str, max_tokens: int = 512):
    """Process a single text file into chunks."""
    if not source_path.exists():
        print(f"File not found: {source_path}")
        return
    
    output_path = output_dir / f"{prefix}_chunks.jsonl"
    
    print(f"Processing {source_path.name}...")
    
    raw_text = read_with_fallback(source_path)
    cleaned = clean_text(raw_text)
    normalized = normalize_text(cleaned)
    sentences = split_sentences(normalized)
    tokenizer = get_tokenizer()
    chunks = chunk_sentences(sentences, tokenizer, max_tokens=max_tokens)
    write_jsonl(chunks, output_path, source_path.name, prefix)

    print(f"  Tokenizer: {tokenizer.name}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Chunks: {len(chunks)} -> {output_path}")


def main():
    """Process multiple text files."""
    output_dir = Path("data/dataset")
    
    # Define files to process: (source_path, prefix)
    files_to_process = [
        (Path("data/text/地煞七十二变.txt"), "disha72"),
        (Path("data/text/张恨水小说全集（套装共37册）.txt"), "zhang_henshui"),
    ]
    
    for source_path, prefix in files_to_process:
        process_file(source_path, output_dir, prefix, max_tokens=512)
    
    print("\nAll files processed!")


if __name__ == "__main__":
    main()
