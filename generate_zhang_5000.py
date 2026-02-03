"""生成5000条纯张恨水风格数据（跳过已有数据）"""

import json
import random
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# 张恨水风格配置
ZHANG_STYLE = {
    "system": "你是一位专业的文学风格转换助手。你擅长将文本改写为：用词雅致古朴、情节委婉曲折、注重情感细节和人物心理的叙事风格。",
    "style_description": "雅致细腻风格"
}

USER_PROMPT_TEMPLATES = [
    "将这段文本改写为{style}：\n{text}",
    "用{style}重写这段话：\n{text}",
    "请以{style}改写：\n{text}",
]


def call_deepseek_api(styled_text: str, config: dict, retry=3) -> str:
    """调用 DeepSeek API 提取平实剧情"""
    for attempt in range(retry):
        try:
            response = requests.post(
                f"{config['generation']['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config['generation']['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config["generation"]["model"],
                    "messages": [
                        {"role": "system", "content": config["system_prompt"]},
                        {"role": "user", "content": styled_text}
                    ],
                    "temperature": config["generation"]["temperature"],
                    "max_tokens": config["generation"]["max_tokens"]
                },
                timeout=config["generation"]["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API 错误 {response.status_code}: {response.text}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def is_zhang_henshui_source(record: dict) -> bool:
    """判断是否为张恨水源数据"""
    source = record.get("source_file", "") + record.get("dataset", "")
    return "zhang" in source.lower() or "恨水" in source


def generate_sample(styled_text: str, plain_text: str, record_id: str) -> dict:
    """生成训练样本"""
    user_template = random.choice(USER_PROMPT_TEMPLATES)
    user_prompt = user_template.format(
        style=ZHANG_STYLE["style_description"], 
        text=plain_text
    )
    
    return {
        "conversations": [
            {"role": "system", "content": ZHANG_STYLE["system"]},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": styled_text}
        ],
        "_record_id": record_id
    }


def process_single_sample(record, api_config):
    """处理单个样本"""
    styled_text = record["text"]
    record_id = record['_id']
    
    # 调用 API 提取平实剧情
    plain_text = call_deepseek_api(styled_text, api_config)
    
    if plain_text:
        sample = generate_sample(styled_text, plain_text, record_id)
        return sample, record_id
    else:
        return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000, help="生成样本总数")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数")
    parser.add_argument("--output", type=str, default="data/stage2_zhang_5000.jsonl", help="输出文件")
    args = parser.parse_args()
    
    # 加载配置
    api_config = json.load(open("data_prep/llm_config.json", encoding="utf-8"))
    
    # 1. 读取现有数据，提取已用的record_id
    print("正在读取现有数据...")
    existing_file = Path("data/stage2_sample_5000.jsonl")
    used_ids = set()
    
    if existing_file.exists():
        with open(existing_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    # 只统计张恨水风格的
                    system_content = sample['conversations'][0]['content']
                    if '雅致古朴' in system_content or '张恨水' in system_content:
                        record_id = sample.get('_record_id')
                        if record_id:
                            used_ids.add(record_id)
                except:
                    continue
        print(f"✓ 已有张恨水样本: {len(used_ids)} 条")
    
    # 2. 读取源数据（只要张恨水相关的）
    print("\n正在读取源数据...")
    input_file = Path("data/dataset/combined_dataset_uniform.jsonl")
    if not input_file.exists():
        input_file = Path("data/dataset/combined_dataset.jsonl")
    
    if not input_file.exists():
        print("错误：找不到源数据文件！")
        return
    
    zhang_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line)
                text = record.get("text", "")
                
                # 生成唯一 ID
                record_id = f"{idx}_{hash(text[:100])}"
                
                # 只要张恨水数据
                if not is_zhang_henshui_source(record):
                    continue
                
                # 跳过已使用的
                if record_id in used_ids:
                    continue
                
                # 长度筛选：800-1200字
                if 800 <= len(text) <= 1200:
                    record['_id'] = record_id
                    zhang_records.append(record)
            except:
                continue
    
    print(f"✓ 可用张恨水数据: {len(zhang_records)} 条")
    print(f"✓ 需要新生成: {args.num_samples - len(used_ids)} 条")
    
    # 3. 计算需要采样的数量
    need_new = args.num_samples - len(used_ids)
    if need_new <= 0:
        print("已有数据足够，无需生成新数据")
        return
    
    if len(zhang_records) < need_new:
        print(f"⚠️  可用数据不足: 需要 {need_new}，实际 {len(zhang_records)}")
        need_new = len(zhang_records)
    
    # 4. 随机采样
    samples_to_process = random.sample(zhang_records, need_new)
    print(f"\n开始生成 {len(samples_to_process)} 条新样本...")
    
    # 5. 准备输出文件
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 先复制已有的张恨水数据
    print("\n复制已有数据...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        if existing_file.exists():
            with open(existing_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        sample = json.loads(line)
                        system_content = sample['conversations'][0]['content']
                        if '雅致古朴' in system_content or '张恨水' in system_content:
                            f_out.write(line)
                    except:
                        continue
    
    print(f"✓ 已复制 {len(used_ids)} 条现有数据")
    
    # 6. 并发生成新数据
    print(f"\n并发生成新数据 (线程数: {args.workers})...")
    new_used_ids = []
    success_count = 0
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_sample, record, api_config): record
                for record in samples_to_process
            }
            
            with tqdm(total=len(samples_to_process), desc="生成样本") as pbar:
                for future in as_completed(futures):
                    sample, record_id = future.result()
                    
                    if sample:
                        # 立即写入文件
                        f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                        new_used_ids.append(record_id)
                        success_count += 1
                    
                    pbar.update(1)
    
    # 7. 更新已使用ID文件
    used_ids_file = Path("data/stage2_used_ids.txt")
    with open(used_ids_file, 'a', encoding='utf-8') as f:
        for record_id in new_used_ids:
            f.write(f"{record_id}\n")
    
    print(f"\n✅ 生成完成！")
    print(f"   已有数据: {len(used_ids)} 条")
    print(f"   新增数据: {success_count} 条")
    print(f"   总计: {len(used_ids) + success_count} 条")
    print(f"   输出文件: {output_file}")


if __name__ == "__main__":
    main()
