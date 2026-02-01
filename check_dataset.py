import json

with open('data/dataset/combined_dataset_uniform.jsonl', encoding='utf-8') as f:
    lines = f.readlines()

print(f'总条数: {len(lines)}')

sample = json.loads(lines[0])
print(f'字段: {list(sample.keys())}')
print(f'第一条文本长度: {len(sample["text"])} 字符')
print(f'第一条文本前100字: {sample["text"][:100]}')

total_chars = sum(len(json.loads(l)["text"]) for l in lines[:1000])
print(f'前1000条平均长度: {total_chars / 1000:.0f} 字符')
