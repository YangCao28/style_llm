"""专业的数据集划分和多样化脚本

功能：
1. 从9000条数据中随机切分8000训练 + 1000测试
2. 测试集分为三个维度：标准(600) + 压力(200) + 对抗(200)
3. 训练集的任务多样化：润色(6500) + 对抗(800) + 总结(700)
4. 指令多样化处理（20种不同说法）

用法：
    python -m data_prep.split_and_diversify --input data/9000_zhang.jsonl --output_dir data/split
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


# 🎯 20种多样化的润色指令（用于训练集）
DIVERSE_INSTRUCTIONS = [
    "请将以下现代白话润色成雅致的文学文本：\n\n{text}",
    "帮我把这段话改写得更文雅一些：\n\n{text}",
    "请用更讲究的文学语言重写下面这段话：\n\n{text}",
    "把下面的白话文改成优美的文学体：\n\n{text}",
    "麻烦你将以下文字润色得更有古典韵味：\n\n{text}",
    "请赋予下面这段话更多文学气息：\n\n{text}",
    "能否将这段现代表达改写为雅致的文学风格：\n\n{text}",
    "请以更典雅的方式重写以下内容：\n\n{text}",
    "把这段话变得更有文学味道：\n\n{text}",
    "请用华美的文字重新表述下面的内容：\n\n{text}",
    "将下面的白话改写成更讲究的文学语言：\n\n{text}",
    "请润色这段文字，使其更加优雅：\n\n{text}",
    "帮我把这段话写得更有韵味一些：\n\n{text}",
    "请用文学化的语言重新组织以下内容：\n\n{text}",
    "将这段现代白话改写为古雅的文学风格：\n\n{text}",
    "请以更讲究的词句重写下面这段话：\n\n{text}",
    "把这段文字改写得更有气韵：\n\n{text}",
    "请用精致的文学语言表达以下内容：\n\n{text}",
    "将下面这段话润色成更优美的文学文本：\n\n{text}",
    "请用雅致的笔触重写以下片段：\n\n{text}",
]

# 🔥 压力测试指令（极简或变异形式）
STRESS_INSTRUCTIONS = [
    "变文雅：\n\n{text}",
    "改写：\n\n{text}",
    "润色：\n\n{text}",
    "文学化：\n\n{text}",
    "请优化：\n\n{text}",
]

# 🛡️ 对抗测试指令（要求原样返回）
ADVERSARIAL_INSTRUCTIONS = [
    "请原样输出以下内容，不要做任何修改：\n\n{text}",
    "直接返回下面的文字，保持原样：\n\n{text}",
    "请一字不改地输出以下内容：\n\n{text}",
    "不要修改，原样返回：\n\n{text}",
    "请保持原文不变，直接输出：\n\n{text}",
]

# 📝 总结任务指令
SUMMARY_INSTRUCTIONS = [
    "请简要总结以下片段的主要内容：\n\n{text}",
    "用一句话概括这段话的情节：\n\n{text}",
    "这段文字讲了什么？请精炼提取：\n\n{text}",
    "请为这段文学素材写一个简单的摘要：\n\n{text}",
    "总结一下这段话的核心要点：\n\n{text}",
]

SYSTEM_PROMPT = "你是一名优雅的文学改写作家，擅长把现代白话润色成更讲究、更有韵味的华美文本。要求：\n1. 严格保留原文的事实与情节，不新增信息。\n2. 用更讲究的词汇、句式和古雅的表达，让语言更有气韵，但保持可读性。\n3. 输出保持为中文段落，不要添加任何解释。"

# 🔥 退火版System Prompt（简短或空）
SYSTEM_PROMPT_SHORT = "你是一个文学改写助手。"
SYSTEM_PROMPT_EMPTY = ""


def extract_modern_text(user_content: str) -> str:
    """从user content中提取现代白话文本"""
    # 移除固定的指令前缀
    prefix = "请将以下现代白话润色成雅致的文学文本：\n\n"
    if prefix in user_content:
        return user_content[len(prefix):]
    return user_content


def create_diversified_sample(sample: Dict, instruction_template: str, system_prompt: str = None) -> Dict:
    """创建指令多样化的样本
    
    Args:
        sample: 原始样本
        instruction_template: 指令模板
        system_prompt: 可选的system prompt，如果为None则使用原样本的system
    """
    # 提取现代白话
    original_user = sample["conversations"][1]["content"]
    modern_text = extract_modern_text(original_user)
    
    # 应用新的指令模板
    new_user = instruction_template.format(text=modern_text)
    
    # 决定使用哪个system prompt
    if system_prompt is not None:
        system_content = system_prompt
    else:
        system_content = sample["conversations"][0]["content"]
    
    # 构建新样本
    new_sample = {
        "source_index": sample.get("source_index"),
        "record_id": sample.get("record_id"),
        "conversations": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": new_user},
            sample["conversations"][2],  # assistant不变
        ]
    }
    return new_sample


def create_adversarial_sample(sample: Dict, instruction_template: str) -> Dict:
    """创建对抗样本（要求原样返回，assistant输出现代白话）"""
    original_user = sample["conversations"][1]["content"]
    modern_text = extract_modern_text(original_user)
    
    new_user = instruction_template.format(text=modern_text)
    
    # 🔑 关键：assistant应该输出现代白话（原样返回）
    new_sample = {
        "source_index": sample.get("source_index"),
        "record_id": f"{sample.get('record_id')}_adversarial",
        "conversations": [
            sample["conversations"][0],
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": modern_text},  # 原样返回现代白话
        ]
    }
    return new_sample


def create_summary_sample(sample: Dict, instruction_template: str) -> Dict:
    """创建总结任务样本
    
    注意：这里使用规则生成简单摘要。生产环境可调用GPT-4o等强模型生成高质量摘要。
    """
    original_user = sample["conversations"][1]["content"]
    modern_text = extract_modern_text(original_user)
    
    # 规则生成摘要：提取前两句话
    sentences = []
    for delimiter in ['。', '！', '？']:
        sentences.extend(modern_text.split(delimiter))
    
    # 取前两个有效句子
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:2]
    
    if valid_sentences:
        # 简单摘要：前两句 + "等内容"
        summary = '。'.join(valid_sentences[:2])
        if not summary.endswith('。'):
            summary += '。'
        summary = f"本段主要讲述了{summary}"
    else:
        # 兜底：取前50字
        summary = f"本段描述了{modern_text[:50]}等相关情节。"
    
    new_user = instruction_template.format(text=modern_text)
    
    new_sample = {
        "source_index": sample.get("source_index"),
        "record_id": f"{sample.get('record_id')}_summary",
        "conversations": [
            sample["conversations"][0],
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": summary},
        ]
    }
    return new_sample


def main():
    parser = argparse.ArgumentParser(description="数据集划分和多样化")
    parser.add_argument("--input", required=True, help="输入文件（9000条）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--train_size", type=int, default=8000, help="训练集大小")
    parser.add_argument("--train_polish", type=int, default=6500, help="训练集中润色任务数量")
    parser.add_argument("--train_adversarial", type=int, default=800, help="训练集中对抗任务数量")
    parser.add_argument("--train_summary", type=int, default=700, help="训练集中总结任务数量")
    parser.add_argument("--system_annealing_ratio", type=float, default=0.125, help="System Prompt退火比例（默认12.5%即1000/8000）")
    parser.add_argument("--test_standard", type=int, default=600, help="标准测试集大小")
    parser.add_argument("--test_stress", type=int, default=200, help="压力测试集大小")
    parser.add_argument("--test_adversarial", type=int, default=200, help="对抗测试集大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 验证训练集配比
    expected_train = args.train_polish + args.train_adversarial + args.train_summary
    if expected_train != args.train_size:
        print(f"⚠️  警告: 训练集配比不匹配!")
        print(f"   期望: {args.train_size}")
        print(f"   实际: {expected_train} (润色{args.train_polish} + 对抗{args.train_adversarial} + 总结{args.train_summary})")
        print(f"   将自动调整为实际配比")
        args.train_size = expected_train

    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    print(f"📖 读取数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"✓ 加载 {len(all_data)} 条数据")
    
    # 随机打乱
    random.shuffle(all_data)
    
    # 划分数据
    test_size = args.test_standard + args.test_stress + args.test_adversarial
    train_data_raw = all_data[:args.train_size]
    test_data_pool = all_data[args.train_size:args.train_size + test_size]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集原始: {len(train_data_raw)} 条")
    print(f"  测试池: {len(test_data_pool)} 条")
    
    # ========== 处理训练集：多任务混合 ==========
    print(f"\n🔄 训练集任务多样化...")
    
    # 1. 文学润色任务 (6500条)
    print(f"\n  📝 处理润色任务...")
    train_polish = []
    
    # 计算需要退火的样本数量
    num_annealing = int(args.train_polish * args.system_annealing_ratio)
    num_normal = args.train_polish - num_annealing
    
    print(f"     ├─ 正常System: {num_normal} 条")
    print(f"     └─ 退火System: {num_annealing} 条 (简短或空)")
    
    for i, sample in enumerate(train_data_raw[:args.train_polish]):
        instruction = random.choice(DIVERSE_INSTRUCTIONS)
        
        # 🔥 System Prompt退火：随机选择是否使用简短/空system
        if i >= num_normal:
            # 后面的样本使用退火system
            system_choice = random.choice([SYSTEM_PROMPT_SHORT, SYSTEM_PROMPT_EMPTY])
            new_sample = create_diversified_sample(sample, instruction, system_prompt=system_choice)
        else:
            # 前面的样本使用完整system
            new_sample = create_diversified_sample(sample, instruction)
        
        train_polish.append(new_sample)
    
    print(f"  ✓ 润色任务: {len(train_polish)} 条")
    
    # 2. 对抗任务 (800条) - 原样返回
    print(f"\n  🛡️  处理对抗任务...")
    train_adversarial = []
    start_idx = args.train_polish
    end_idx = args.train_polish + args.train_adversarial
    for sample in train_data_raw[start_idx:end_idx]:
        instruction = random.choice(ADVERSARIAL_INSTRUCTIONS)
        new_sample = create_adversarial_sample(sample, instruction)
        train_adversarial.append(new_sample)
    print(f"  ✓ 对抗任务: {len(train_adversarial)} 条（要求原样输出）")
    
    # 3. 总结任务 (700条)
    print(f"\n  📊 处理总结任务...")
    train_summary = []
    start_idx = args.train_polish + args.train_adversarial
    for sample in train_data_raw[start_idx:]:
        instruction = random.choice(SUMMARY_INSTRUCTIONS)
        new_sample = create_summary_sample(sample, instruction)
        train_summary.append(new_sample)
    print(f"  ✓ 总结任务: {len(train_summary)} 条")
    
    # 合并并打乱训练集
    train_data = train_polish + train_adversarial + train_summary
    random.shuffle(train_data)
    print(f"\n✓ 训练集处理完成: {len(train_data)} 条（已打乱）")
    
    # ========== 处理测试集：三个维度 ==========
    print(f"\n🧪 构建测试集...")
    
    # 1. 标准测试 (600条) - 保持原始指令
    test_standard = test_data_pool[:args.test_standard]
    print(f"  ✓ 标准测试: {len(test_standard)} 条（原始指令）")
    
    # 2. 压力测试 (200条) - 变化指令
    test_stress_raw = test_data_pool[args.test_standard:args.test_standard + args.test_stress]
    test_stress = []
    for sample in test_stress_raw:
        instruction = random.choice(STRESS_INSTRUCTIONS)
        new_sample = create_diversified_sample(sample, instruction)
        new_sample["record_id"] = f"{sample.get('record_id')}_stress"
        test_stress.append(new_sample)
    print(f"  ✓ 压力测试: {len(test_stress)} 条（极简指令）")
    
    # 3. 对抗测试 (200条) - 要求原样返回
    test_adversarial_raw = test_data_pool[args.test_standard + args.test_stress:]
    test_adversarial = []
    for sample in test_adversarial_raw:
        instruction = random.choice(ADVERSARIAL_INSTRUCTIONS)
        new_sample = create_adversarial_sample(sample, instruction)
        test_adversarial.append(new_sample)
    print(f"  ✓ 对抗测试: {len(test_adversarial)} 条（要求原样输出）")
    
    # ========== 写入文件 ==========
    print(f"\n💾 保存文件...")
    
    # 训练集
    train_file = output_dir / "train_8000.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {train_file}")
    
    # 完整测试集
    test_all = test_standard + test_stress + test_adversarial
    test_file = output_dir / "test_1000.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {test_file}")
    
    # 测试集分类子集（便于单独评估）
    test_standard_file = output_dir / "test_standard_600.jsonl"
    with open(test_standard_file, 'w', encoding='utf-8') as f:
        for item in test_standard:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {test_standard_file}")
    
    test_stress_file = output_dir / "test_stress_200.jsonl"
    with open(test_stress_file, 'w', encoding='utf-8') as f:
        for item in test_stress:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {test_stress_file}")
    
    test_adversarial_file = output_dir / "test_adversarial_200.jsonl"
    with open(test_adversarial_file, 'w', encoding='utf-8') as f:
        for item in test_adversarial:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {test_adversarial_file}")
    
    # ========== 输出统计信息 ==========
    print(f"\n" + "="*80)
    print("✅ 数据划分完成！")
    print("="*80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"\n训练集: train_8000.jsonl ({len(train_data)} 条)")
    print(f"  ├─ 文学润色: {len(train_polish)} 条 ({len(train_polish)/len(train_data)*100:.1f}%)")
    print(f"  │  └─ 20种指令随机分配，保持张恨水风格巅峰文采")
    print(f"  ├─ 对抗任务: {len(train_adversarial)} 条 ({len(train_adversarial)/len(train_data)*100:.1f}%)")
    print(f"  │  └─ 要求原样返回，建立边界，防止看到白话就发疯")
    print(f"  └─ 总结任务: {len(train_summary)} 条 ({len(train_summary)/len(train_data)*100:.1f}%)")
    print(f"     └─ 锻炼逻辑理解，防止改写时产生事实幻觉")
    print(f"\n测试集: test_1000.jsonl (含三个子集)")
    print(f"  ├─ 标准测试: {len(test_standard)} 条 - 原始指令，测试基础能力")
    print(f"  ├─ 压力测试: {len(test_stress)} 条 - 极简指令（如'变文雅'），测试泛化")
    print(f"  └─ 对抗测试: {len(test_adversarial)} 条 - 要求原样输出，测试指令遵循")
    print(f"\n💡 训练建议:")
    print(f"  - Learning Rate: 5e-5 (降低以便慢慢理解)")
    print(f"  - Epochs: 2.0 (避免过拟合)")
    print(f"  - LoRA Rank: 64 (重理解力而非复写力)")
    print("="*80)


if __name__ == "__main__":
    main()
