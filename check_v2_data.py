import json
from collections import Counter

# 检查v2数据的质量
with open('data/stage2_sample_5000_v2.jsonl', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

print(f"总样本数: {len(samples)}")

# 统计所有system prompt
system_prompts = [s['conversations'][0]['content'] for s in samples]
prompt_counts = Counter(system_prompts)

print(f"\n风格分布:")
for prompt, count in prompt_counts.most_common():
    print(f"  {count:4d}条: {prompt[:100]}...")

# 检查是否还有作家名
ni_kuang_count = sum(1 for p in system_prompts if '倪匡' in p)
zhang_count = sum(1 for p in system_prompts if '张恨水' in p)

print(f"\n包含'倪匡': {ni_kuang_count}条")
print(f"包含'张恨水': {zhang_count}条")

# 检查user prompt中的描述
print(f"\n前5个样本的user prompt关键词:")
for i, sample in enumerate(samples[:5]):
    user_content = sample['conversations'][1]['content']
    print(f"\n样本{i+1}:")
    print(f"  User前80字: {user_content[:80]}...")
    
    # 检查是否有作家名
    if '倪匡' in user_content or '张恨水' in user_content:
        print(f"    ⚠️  User prompt包含作家名！")

# 检查assistant回复质量
print(f"\n\n前3个样本的完整对话:")
for i, sample in enumerate(samples[:3]):
    print(f"\n{'='*80}")
    print(f"样本{i+1}:")
    print(f"{'='*80}")
    for msg in sample['conversations']:
        role = msg['role']
        content = msg['content']
        print(f"\n[{role.upper()}]:")
        print(content[:500] if len(content) > 500 else content)
        if len(content) > 500:
            print(f"...(还有{len(content)-500}字符)")

# 检查是否有娱乐化内容
print(f"\n\n检查娱乐化标记:")
bad_markers = ['哈哈', '嘻嘻', '😉', '~~~', '。。。', '咯~', '哦~', '喽~', '哈哒哒', '表情包']
for i, sample in enumerate(samples[:50]):
    assistant_content = sample['conversations'][2]['content']
    found = [m for m in bad_markers if m in assistant_content]
    if found:
        print(f"  样本{i+1}: {found}")
