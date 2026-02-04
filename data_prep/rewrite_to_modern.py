"""用DeepSeek将张恨水文本改写为简单现代白话文

支持断点续传功能
"""

import argparse
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# System prompt：指导API改写为现代白话
REWRITE_SYSTEM = """# Role
你是一个初级的、缺乏文学素养的人工智能助手。

# Task
将提供的【文学原著】改写成一段典型的【AI 生成式白话】。

# Rules (核心风格指南)
1. **典型的 AI 翻译腔**：
   - 多用"被"、"关于"、"进行"、"作出"等虚词。
   - 句式要长且死板（例：将"他骂着走了"改为"他带着愤怒的情绪离开了现场"）。
2. **逻辑列表化（可选）**：
   - 尝试用"首先、其次、最后"或者"第一、第二"来拆解原著的情节。
3. **词汇贫乏且重复**：
   - 重复使用"非常"、"特别"、"表现出"、"显示了"等万金油词汇。
   - 绝对禁止任何文学意象，只准描述事实。
4. **过度解释（机械感）**：
   - 像说明书一样解释原著中的动作（例：将"一揖到地"改为"做出了一个向下弯腰 90 度的身体动作以示尊重"）。

# Output Format
直接输出那段充满"机械味"和"翻译腔"的白话文，不要任何解释。"""

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
    parser.add_argument("--output", default="data/modern_pairs_chunks", help="输出目录（存储分块文件）")
    parser.add_argument("--count", type=int, default=None, help="生成样本数量（None=全部）")
    parser.add_argument("--chunk_size", type=int, default=100, help="每个分块文件的大小")
    parser.add_argument("--workers", type=int, default=10, help="并发请求数量")
    parser.add_argument("--start_from", type=int, default=0, help="从第几个样本开始处理")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载API配置
    api_config = json.load(open("data_prep/llm_config.json", encoding="utf-8"))

    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"总记录数: {len(records)}")
    
    # 确定要处理的样本数量
    if args.count is None:
        sample_size = len(records)
    else:
        sample_size = min(args.count, len(records))
    
    # 检查已有的分块文件，确定从哪里继续
    existing_chunks = sorted(output_dir.glob("chunk_*.jsonl"))
    if existing_chunks and args.start_from == 0:
        last_chunk = existing_chunks[-1]
        # 从文件名提取最后一个chunk的编号（chunk_0000.jsonl -> 0）
        last_chunk_num = int(last_chunk.stem.split("_")[1])
        args.start_from = (last_chunk_num + 1) * args.chunk_size
        print(f"📂 检测到已有 {len(existing_chunks)} 个分块文件")
        print(f"✓ 从索引 {args.start_from} 继续处理")
    
    if args.start_from >= sample_size:
        print(f"\n✅ 所有样本已完成！")
        return
    
    print(f"\n📊 任务统计:")
    print(f"  目标样本数: {sample_size}")
    print(f"  起始位置: {args.start_from}")
    print(f"  剩余样本: {sample_size - args.start_from}")
    print(f"  分块大小: {args.chunk_size}")
    print(f"  并发数: {args.workers}")
    print(f"\n🚀 开始处理...\n")

    def process_record(idx: int, record: dict):
        original_text = record['text']
        modern_text = call_deepseek(original_text, api_config)
        return {
            "index": idx,
            "record_id": record.get("_id", f"zhang_clean_{idx}"),
            "original": original_text,
            "modern": modern_text
        }

    futures = {}
    results_dict = {}  # key=索引，value=结果项
    chunk_num = args.start_from // args.chunk_size
    
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        # 提交所有任务
        for idx in range(args.start_from, sample_size):
            futures[executor.submit(process_record, idx, records[idx])] = idx

        # 处理完成的任务
        for future in as_completed(futures):
            idx = futures[future]
            sample_no = idx + 1
            
            print(f"处理样本 {sample_no}/{sample_size} (idx={idx})...")
            
            try:
                result = future.result()
            except Exception as exc:
                print(f"✗ 发生异常: {exc}\n")
                continue

            original_text = result["original"]
            modern_text = result["modern"]

            if modern_text:
                print(f"✓ 改写成功")
                print(f"  原文片段: {original_text[:60]}...")
                print(f"  现代文片段: {modern_text[:60]}...")
                print(f"  长度对比: {len(original_text)} → {len(modern_text)} 字 ({len(modern_text)/len(original_text):.1%})")

                finetune_user_prompt = TRAIN_USER_TEMPLATE.format(text=modern_text)
                item = {
                    "source_index": result["index"],
                    "record_id": result["record_id"],
                    "conversations": [
                        {"role": "system", "content": TRAIN_SYSTEM},
                        {"role": "user", "content": finetune_user_prompt},
                        {"role": "assistant", "content": original_text}
                    ]
                }
                
                results_dict[result["index"]] = item
                
                # 🔑 每累积chunk_size个结果就保存一个chunk
                if len(results_dict) >= args.chunk_size:
                    # 找出当前字典中最小和最大的索引，确定chunk编号
                    min_idx = min(results_dict.keys())
                    current_chunk = min_idx // args.chunk_size
                    
                    chunk_file = output_dir / f"chunk_{current_chunk:04d}.jsonl"
                    
                    # 按索引排序后写入
                    sorted_indices = sorted(results_dict.keys())
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        for i in sorted_indices:
                            f.write(json.dumps(results_dict[i], ensure_ascii=False) + "\n")
                    
                    print(f"💾 保存分块文件: {chunk_file} ({len(results_dict)} 个样本)\n")
                    results_dict.clear()
                    chunk_num = current_chunk + 1
            else:
                print(f"✗ 改写失败\n")
    
    # 保存最后剩余的样本
    if results_dict:
        min_idx = min(results_dict.keys())
        current_chunk = min_idx // args.chunk_size
        chunk_file = output_dir / f"chunk_{current_chunk:04d}.jsonl"
        
        sorted_indices = sorted(results_dict.keys())
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for i in sorted_indices:
                f.write(json.dumps(results_dict[i], ensure_ascii=False) + "\n")
        print(f"💾 保存最后的分块文件: {chunk_file} ({len(results_dict)} 个样本)")
    
    print(f"\n✅ 完成！")
    print(f"   输出目录: {output_dir}")
    print(f"\n💡 下一步：运行合并脚本")
    print(f"   python -m data_prep.merge_chunks --input {output_dir} --output data/modern_pairs_final.jsonl")
    
    # 显示完整的第一个样本
    if results_dict:
        print(f"\n{'='*80}")
        print(f"样本示例 (索引0):")
        print(f"{'='*80}")
        if 0 in results_dict:
            first = results_dict[0]
            conv = first['conversations']
            print(f"\n【记录ID】: {first['record_id']}")
            for message in conv:
                role = message['role']
                header = {
                    'system': 'System prompt',
                    'user': 'User prompt',
                    'assistant': 'Assistant输出'
                }[role]
                print(f"\n【{header}】:\n{message['content'][:200]}...")


if __name__ == "__main__":
    main()
