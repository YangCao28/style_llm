import json
import os

# Configuration
input_file = 'data/wuxia_vernacular_pairs.jsonl'
output_file = 'data/wuxia_sft_data.json'

system_prompt = "你是一位通晓古今的武侠小说大师，继承了金庸、古龙的笔法。你的任务是将现代白话文改写为地道的武侠风格。要求：用词古雅，多用四字成语，描写需体现江湖气息，注重动作的凌厉与意境的苍凉。"

data = []

if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} not found.")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        vernacular = item.get('vernacular_text', '').strip()
        wuxia = item.get('wuxia_text', '').strip()
        
        if vernacular and wuxia:
            entry = {
                "conversations": [
                    {
                        "from": "system",
                        "value": system_prompt
                    },
                    {
                        "from": "human",
                        "value": f"把这句话改成武侠风：{vernacular}"
                    },
                    {
                        "from": "gpt",
                        "value": wuxia
                    }
                ]
            }
            data.append(entry)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Successfully converted {len(data)} items to {output_file}")
