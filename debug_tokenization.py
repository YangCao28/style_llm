from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained('stage1_style_injection/checkpoint-531', trust_remote_code=True)
data = [json.loads(l) for l in open('data/stage2_sample_5000.jsonl', encoding='utf-8')]
example = data[0]

conversations = example['conversations']
messages = []
assistant_response = None

for msg in conversations:
    role = msg['role']
    content = msg['content']
    
    if role == 'system':
        norm_role = 'system'
    elif role == 'assistant':
        norm_role = 'assistant'
        assistant_response = content
    else:
        norm_role = 'user'
    
    messages.append({'role': norm_role, 'content': content})

prompt_messages = [m for m in messages if m['role'] != 'assistant']
prompt_parts = []
for msg in prompt_messages:
    prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
prompt_parts.append('<|im_start|>assistant\n')
prompt_text = '\n'.join(prompt_parts)

full_text = prompt_text + assistant_response + '<|im_end|>'

prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
full_ids = tokenizer(full_text, truncation=True, max_length=2048, add_special_tokens=False)['input_ids']

labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

print(f'Prompt length: {len(prompt_ids)}')
print(f'Full length: {len(full_ids)}')
print(f'Assistant length: {len(full_ids) - len(prompt_ids)}')
print(f'Label mask length (trainable tokens): {len([i for i in labels if i != -100])}')
print(f'\nFirst 10 prompt tokens: {tokenizer.convert_ids_to_tokens(prompt_ids[:10])}')
print(f'Last 10 tokens: {tokenizer.convert_ids_to_tokens(full_ids[-10:])}')
