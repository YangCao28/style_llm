"""用DeepSeek将张恨水文本改写为简单现代白话文"""

import argparse
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# System prompt：指导API改写为现代白话
REWRITE_SYSTEM = """你是一个专业的文本改写助手。你的任务是：
1. 将提供的民国时期文学文本改写为简单的现代白话文，要求文盲也能读懂
2. 保留所有关键信息（人物、事件、对话、情节）
3. 将古雅词汇替换为现代常用词
4. 将长句拆分为短句，使表达更简洁明了
5. 去掉修饰，去掉描写，力求简洁直白，保持叙事清晰
6. 不添加任何原文没有的内容

输出格式：直接给出改写后的现代白话文，不要有任何说明或评论。"""

USER_TEMPLATE = "请将以下文本改写为简单的现代白话文，要求文盲也能读懂，去掉描写：\n\n{text}"

# System prompt：用于未来fine-tuning，指导模型将白话文变得雅致
TRAIN_SYSTEM = """你是一名优雅的文学改写作家，擅长把现代白话润色成更讲究、更有韵味的华美文本。要求：
1. 严格保留原文的事实与情节，不新增信息。
2. 用更讲究的词汇、句式和古雅的表达，让语言更有气韵，但保持可读性。
3. 输出保持为中文段落，不要添加任何解释。"""

TRAIN_USER_TEMPLATE = "请将以下现代白话润色成雅致的文学文本：\n\n{text}"


def call_deepseek(text: str, api_config: dict, retry=3):
    """调用DeepSeek API改写文本"""
    for attempt in range(retry):
        try:
            response = requests.post(
                f"{api_config['generation']['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_config['generation']['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": api_config["generation"]["model"],
                    "messages": [
                        {"role": "system", "content": REWRITE_SYSTEM},
                        {"role": "user", "content": USER_TEMPLATE.format(text=text)}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API错误 {response.status_code}: {response.text}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="批量将张恨水文本改写为现代白话文")
    parser.add_argument("--input", default="data/dataset/zhang_cleaned_new.jsonl", help="输入JSONL文件路径")
    parser.add_argument("--output", default="data/modern_pairs_10.jsonl", help="输出JSONL文件路径")
    parser.add_argument("--count", type=int, default=10, help="生成样本数量")
    parser.add_argument("--workers", type=int, default=1, help="并发请求数量")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    # 加载API配置
    api_config = json.load(open("data_prep/llm_config.json", encoding="utf-8"))

    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"总记录数: {len(records)}")
    sample_size = min(args.count, len(records))
    print(f"\n生成前{sample_size}个样本，使用 {args.workers} 个并发worker...\n")

    def process_record(idx: int, record: dict):
        original_text = record['text']
        modern_text = call_deepseek(original_text, api_config)
        return {
            "index": idx,
            "record_id": record.get("_id", f"zhang_{idx}"),
            "original": original_text,
            "modern": modern_text
        }

    futures = {}
    results = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for i in range(sample_size):
            futures[executor.submit(process_record, i, records[i])] = i

        for future in as_completed(futures):
            idx = futures[future]
            sample_no = idx + 1
            try:
                result = future.result()
            except Exception as exc:
                print(f"✗ 样本 {sample_no}/{sample_size} 发生异常: {exc}")
                continue

            original_text = result["original"]
            modern_text = result["modern"]

            if modern_text:
                print(f"✓ 样本 {sample_no}/{sample_size} 改写成功")
                print(f"  原文片段: {original_text[:60]}...")
                print(f"  现代文片段: {modern_text[:60]}...")
                print(f"  长度对比: {len(original_text)} → {len(modern_text)} 字 ({len(modern_text)/len(original_text):.1%})")

                finetune_user_prompt = TRAIN_USER_TEMPLATE.format(text=modern_text)
                results.append((
                    result["index"],
                    {
                        "record_id": result["record_id"],
                        "conversations": [
                            {"role": "system", "content": TRAIN_SYSTEM},
                            {"role": "user", "content": finetune_user_prompt},
                            {"role": "assistant", "content": original_text}
                        ]
                    }
                ))
            else:
                print(f"✗ 样本 {sample_no}/{sample_size} 改写失败")

            print()
    
    # 保存结果
    results.sort(key=lambda x: x[0])
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 完成！生成了 {len(results)} 个样本")
    print(f"   输出文件: {output_file}")
    
    # 显示完整的第一个样本
    if results:
        print(f"\n{'='*80}")
        print(f"完整样本1:")
        print(f"{'='*80}")
        _, first = results[0]
        conv = first['conversations']
        print(f"\n【记录ID】: {first['record_id']}")
        for message in conv:
            role = message['role']
            header = {
                'system': 'System prompt',
                'user': 'User prompt',
                'assistant': 'Assistant输出'
            }[role]
            print(f"\n【{header}】:\n{message['content']}")


if __name__ == "__main__":
    main()
