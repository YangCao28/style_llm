"""快速生成 Stage 2 样本用于测试"""

import json
import random
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# 风格配置 - 使用通用文学特征描述
STYLE_CONFIG = {
    "倪匡": {
        "system": "你是一位专业的文学风格转换助手。你擅长将文本改写为：语言简洁明快、逻辑严密、节奏紧凑、善于营造氛围的叙事风格。",
        "keywords": ["简洁", "紧凑", "氛围"],
        "dataset_pattern": r"disha.*72",
        "style_description": "简洁紧凑风格"
    },
    "张恨水": {
        "system": "你是一位专业的文学风格转换助手。你擅长将文本改写为：用词雅致古朴、情节委婉曲折、注重情感细节和人物心理的叙事风格。",
        "keywords": ["雅致", "委婉", "细腻"],
        "dataset_pattern": r"zhang.*henshui",
        "style_description": "雅致细腻风格"
    },
}

USER_PROMPT_TEMPLATES = [
    "将这段文本改写为{style}：\n{text}",
    "用{style}重写这段话：\n{text}",
    "请以{style}改写：\n{text}",
]


def call_deepseek_api(styled_text: str, config: dict, retry=3) -> str:
    """调用 DeepSeek API 总结剧情要点，带重试机制"""
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
                        {"role": "user", "content": styled_text}  # 发送完整文本
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
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                return None
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def classify_style(record: dict) -> str:
    """判断文本属于哪个风格"""
    source = record.get("source_file", "") + record.get("dataset", "")
    
    if "disha" in source.lower() or "72" in source:
        return "倪匡"
    elif "zhang" in source.lower() or "恨水" in source:
        return "张恨水"
    
    return None


def generate_stage2_sample(styled_text: str, plain_text: str, style: str) -> dict:
    """生成训练样本 - 使用 conversations 格式（兼容 LLaMA Factory 等框架）"""
    config = STYLE_CONFIG[style]
    user_template = random.choice(USER_PROMPT_TEMPLATES)
    # 使用 style_description 而不是 style（作家名字）
    user_prompt = user_template.format(
        style=config.get("style_description", style), 
        text=plain_text
    )
    
    return {
        "conversations": [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": styled_text}
        ]
    }


def process_single_sample(record, style, api_config):
    """处理单个样本（用于并发）"""
    styled_text = record["text"]
    record_id = record['_id']
    
    # 调用 API 总结剧情要点
    plain_text = call_deepseek_api(styled_text, api_config)
    
    if plain_text:
        sample = generate_stage2_sample(styled_text, plain_text, style)
        sample['_record_id'] = record_id
        return sample, record_id
    else:
        return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000, help="生成样本数量")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数")
    parser.add_argument("--batch_size", type=int, default=100, help="每批保存数量")
    parser.add_argument("--output", type=str, default="data/stage2_sample_5000.jsonl", help="输出文件路径")
    args = parser.parse_args()
    
    # 加载配置
    api_config = json.load(open("data_prep/llm_config.json", encoding="utf-8"))
    
    # 加载已使用的数据 ID
    used_ids_file = Path("data/stage2_used_ids.txt")
    used_ids = set()
    if used_ids_file.exists():
        with open(used_ids_file, 'r', encoding='utf-8') as f:
            used_ids = set(line.strip() for line in f if line.strip())
        print(f"已加载 {len(used_ids)} 条已使用的数据 ID")
    
    # 读取数据
    input_file = Path("data/dataset/combined_dataset_uniform.jsonl")
    style_records = {style: [] for style in STYLE_CONFIG.keys()}  # 按风格分类
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line)
                text = record.get("text", "")
                
                # 生成唯一 ID（使用行号 + 文本前100字符的hash）
                record_id = f"{idx}_{hash(text[:100])}"
                
                # 跳过已使用的数据
                if record_id in used_ids:
                    continue
                
                # 严格选择 800-1200 字的完整片段，太长或太短都不要
                if len(text) >= 800 and len(text) <= 1200:
                    style = classify_style(record)
                    if style:
                        record['_id'] = record_id  # 保存 ID
                        style_records[style].append(record)
            except:
                continue
    
    print(f"按风格分类候选数据:")
    for style, records in style_records.items():
        print(f"  {style}: {len(records)} 条")
    
    # 灵活分配策略：优先倪匡，其次张恨水
    total_target = args.num_samples
    
    # 倪匡：尽量40%，不足就全部
    nikuang_target = int(total_target * 0.40)
    nikuang_available = len(style_records.get("倪匡", []))
    nikuang_actual = min(nikuang_target, nikuang_available)
    
    # 剩余名额全部给张恨水
    remaining = total_target - nikuang_actual
    zhanghenshui_available = len(style_records.get("张恨水", []))
    zhanghenshui_actual = min(remaining, zhanghenshui_available)
    
    target_distribution = {
        "倪匡": nikuang_actual,
        "张恨水": zhanghenshui_actual,
    }
    
    print(f"\n目标分配:")
    print(f"  倪匡: {nikuang_actual} 条 (可用: {nikuang_available})")
    print(f"  张恨水: {zhanghenshui_actual} 条 (可用: {zhanghenshui_available})")
    print(f"  总计: {sum(target_distribution.values())} 条")
    
    # 从每个风格池中采样
    samples = []
    for style, target_count in target_distribution.items():
        available = style_records[style]
        if len(available) < target_count:
            print(f"⚠️  {style} 数据不足: 需要 {target_count}，实际 {len(available)}")
            sampled = available
        else:
            sampled = random.sample(available, target_count)
        
        for record in sampled:
            samples.append((record, style))
    
    print(f"实际采样: {len(samples)} 条")
    
    # 输出文件
    output_file = Path(args.output)
    summary_file = Path(f"{output_file.stem}_summary.jsonl")
    if output_file.parent != Path("."):
        summary_file = output_file.parent / summary_file
    
    # 打开文件准备追加写入
    f_train = open(output_file, 'w', encoding='utf-8')
    f_summary = open(summary_file, 'w', encoding='utf-8')
    
    print(f"\n开始并发生成 (线程数: {args.workers}, 批次大小: {args.batch_size})...")
    
    new_used_ids = []
    batch_count = 0
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_sample, record, style, api_config): (record, style)
            for record, style in samples
        }
        
        with tqdm(total=len(samples), desc="生成样本") as pbar:
            for future in as_completed(futures):
                sample, record_id = future.result()
                
                if sample:
                    # 立即写入文件（防止丢失）
                    f_train.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    f_train.flush()  # 强制刷新到磁盘
                    
                    # 写入总结文件
                    user_content = sample["conversations"][1]["content"]
                    assistant_content = sample["conversations"][2]["content"]
                    summary_record = {
                        "plain_text": user_content.split("：\n")[-1] if "：\n" in user_content else user_content,
                        "styled_text": assistant_content,
                        "style": sample["conversations"][0]["content"].split("你是")[1].split("，")[0] if "你是" in sample["conversations"][0]["content"] else "unknown",
                        "record_id": sample.get("_record_id", "")
                    }
                    f_summary.write(json.dumps(summary_record, ensure_ascii=False) + '\n')
                    f_summary.flush()
                    
                    new_used_ids.append(record_id)
                    batch_count += 1
                    
                    # 每批次保存已使用ID
                    if batch_count >= args.batch_size:
                        with open(used_ids_file, 'a', encoding='utf-8') as f_ids:
                            for rid in new_used_ids:
                                f_ids.write(f"{rid}\n")
                        new_used_ids = []
                        batch_count = 0
                
                pbar.update(1)
    
    # 关闭文件
    f_train.close()
    f_summary.close()
    
    # 保存剩余的 ID
    if new_used_ids:
        with open(used_ids_file, 'a', encoding='utf-8') as f:
            for record_id in new_used_ids:
                f.write(f"{record_id}\n")
    
    print(f"\n✅ 生成完成！")
    print(f"✅ 训练数据: {output_file}")
    print(f"✅ 平实总结: {summary_file}")


if __name__ == "__main__":
    main()
