import json

# 统计训练数据中user prompt的格式
with open('data/stage2_zhang_5000_cleaned.jsonl', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

# 提取user prompt的第一行（指令部分）
user_prompts = [s['conversations'][1]['content'].split('\n')[0] for s in samples]

# 统计格式
from collections import Counter
prompt_patterns = Counter(user_prompts)

print(f"训练数据中的User Prompt格式分布（前10种）:\n")
for pattern, count in prompt_patterns.most_common(10):
    print(f"{count:4d}条: {pattern}")

print(f"\n总计: {len(samples)} 条")
print(f"格式种类: {len(prompt_patterns)} 种")
