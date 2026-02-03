import json
from collections import Counter

# 检查旧数据的风格分布
with open('data/stage2_sample_5000.jsonl', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

print(f"总样本数: {len(samples)}")

# 统计所有system prompt
system_prompts = [s['conversations'][0]['content'] for s in samples]
prompt_counts = Counter(system_prompts)

print(f"\n风格分布:")
for prompt, count in prompt_counts.most_common():
    print(f"  {count:4d}条: {prompt[:80]}...")

# 检查是否有"倪匡"或"张恨水"
ni_kuang_count = sum(1 for p in system_prompts if '倪匡' in p)
zhang_count = sum(1 for p in system_prompts if '张恨水' in p)

print(f"\n包含'倪匡': {ni_kuang_count}条")
print(f"包含'张恨水': {zhang_count}条")

# 检查assistant回复质量（前5个样本）
print(f"\n前5个样本的assistant回复长度:")
for i, sample in enumerate(samples[:5]):
    assistant_content = sample['conversations'][2]['content']
    print(f"  样本{i+1}: {len(assistant_content)}字符")
    
    # 检查是否有网络用语标记
    bad_markers = ['哈哈', '嘻嘻', '😉', '~~~', '。。。', '咯', '哦~', '喽~']
    found = [m for m in bad_markers if m in assistant_content]
    if found:
        print(f"    ⚠️  包含娱乐化标记: {found}")
