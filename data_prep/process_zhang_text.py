"""用DeepSeek完成三合一任务：改写现代文、清理章回、修复片段"""

import json
import requests
import time
from pathlib import Path

# System prompt：简化任务
TASK_SYSTEM = """你是一个专业的文本简化助手。

**任务**：将给定的文学文本简化为现代白话文。

**输入**：一段文学作品的原文（可能包含古雅表达、章回体标记等）
**输出**：这段原文的现代白话版本

**处理步骤**：

1. **移除无关标记**（在输出中不要包含）：
   - "序言"、"前言"、"后记"、"自序"等段落标题
   - "第X章"、"第X回"及章节标题
   - "下回分解"、"欲知后事如何"、"且听下回分解"
   - "回到首页"、"返回目录"等导航文字
   - "本章完"、"未完待续"

2. **简化叙事开头**（在输出中）：
   - 如果原文句子以"话说"、"却说"、"且说"、"再说"开头，输出时去掉这些词
   - 例如：原文"话说王小姐心中忐忑" → 输出"王小姐心中忐忑"

3. **语言现代化**：
   - 古雅词汇 → 常用现代词
   - 长句 → 短句
   - 文言句式 → 白话句式

**核心原则**：
- ✅ 这是**对同一段内容的改写**，不是重新创作
- ✅ 原文的每个情节、人物、对话、动作必须在输出中体现
- ✅ 不添加原文没有的内容
- ✅ 不省略原文已有的信息
- ✅ 只改变表达方式，不改变内容本身

**输出格式**：直接输出简化后的文本，不要任何解释或说明。"""

USER_TEMPLATE = "请处理以下文本：\n\n{text}"


def call_deepseek(text: str, api_config: dict, retry=3):
    """调用DeepSeek API处理文本"""
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
                        {"role": "system", "content": TASK_SYSTEM},
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="生成样本数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()
    
    # 加载API配置
    api_config = json.load(open("data_prep/llm_config.json", encoding="utf-8"))
    
    # 读取张恨水数据
    input_file = Path("data/dataset/zhang_cleaned.jsonl")
    
    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    
    print(f"总记录数: {len(records)}")
    print(f"\n生成前{args.num}个样本...\n")
    
    # 训练数据的system prompt
    TRAIN_SYSTEM = (
        "你是一个专业的文学改写工具，仅执行文风转换任务。"
        "你的唯一职责是：对给定文本进行文风调整，使其呈现用词雅致古朴、情节委婉曲折、"
        "注重情感细节和人物心理的叙事风格，同时保持内容完全一致。"
        "严格禁止：(1)添加任何原文不存在的信息、情节、人物、对话；(2)删减或略去任何原文信息；"
        "(3)在改写结果后继续生成任何内容；(4)生成任何解释、评论、提示或元信息。"
        "输出格式：直接给出改写后的文本，不包含任何其他内容。改写完成后立即停止生成。"
    )
    
    results = []
    success_count = 0
    
    for i in range(args.num):
        record = records[i]
        original_text = record['text']
        
        print(f"{'='*80}")
        print(f"样本 {i+1}/{args.num}")
        print(f"{'='*80}")
        print(f"原文长度: {len(original_text)} 字")
        print(f"原文前80字: {original_text[:80]}...")
        
        # 调用API处理（生成现代白话文）
        print(f"调用API处理（清理+修复+改写现代文）...")
        modern_text = call_deepseek(original_text, api_config)
        
        if modern_text:
            print(f"✓ 处理成功")
            print(f"现代文长度: {len(modern_text)} 字 ({len(modern_text)/len(original_text):.1%})")
            print(f"现代文前80字: {modern_text[:80]}...")
            
            # 构建训练样本
            # user: 现代白话文（简洁原文）
            # assistant: 张恨水原文（目标风格）
            training_sample = {
                "conversations": [
                    {"role": "system", "content": TRAIN_SYSTEM},
                    {"role": "user", "content": f"用雅致细腻风格的笔法重写这段话：\n{modern_text}"},
                    {"role": "assistant", "content": original_text}
                ],
                "_record_id": record.get("_id", f"zhang_{i}")
            }
            
            results.append(training_sample)
            success_count += 1
        else:
            print(f"✗ 处理失败")
        
        print()
        time.sleep(0.5)  # 避免API限流
    
    # 保存结果为jsonl格式
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"data/stage2_zhang_{args.num}.jsonl")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！成功生成 {success_count}/{args.num} 个训练样本")
    print(f"   输出文件: {output_file}")
    
    # 显示第一个样本
    if results:
        print(f"\n{'='*80}")
        print(f"训练样本1:")
        print(f"{'='*80}")
        sample = results[0]
        print(f"\n[SYSTEM]:")
        print(sample['conversations'][0]['content'][:150] + "...")
        print(f"\n[USER] (现代白话 {len(sample['conversations'][1]['content'])}字):")
        print(sample['conversations'][1]['content'][:200] + "...")
        print(f"\n[ASSISTANT] (张恨水风格 {len(sample['conversations'][2]['content'])}字):")
        print(sample['conversations'][2]['content'][:200] + "...")


if __name__ == "__main__":
    main()
