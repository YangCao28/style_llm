import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    data_path = Path("data/stage2_sample_5000.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    # 用和训练相同的 tokenizer；如果远端用的是别的模型，可以改这里
    tokenizer_name = "Qwen/Qwen3-8B-Base"
    print(f"Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            conv = rec.get("conversations", [])
            text = "\n".join(
                f"<|im_start|>{m['from']}\n{m['value']}<|im_end|>" for m in conv
            )
            ids = tok.encode(text)
            lengths.append(len(ids))

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[int(n * p)]

    print(f"样本数: {n}")
    print(f"中位数 tokens: {pct(0.5)}")
    print(f"90 分位 tokens: {pct(0.9)}")
    print(f"95 分位 tokens: {pct(0.95)}")
    print(f"最大 tokens: {lengths[-1]}")


if __name__ == "__main__":
    main()
