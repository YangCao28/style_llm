"""批量修改prompt：去掉作家名，改为概括性文风描述"""

import json
from pathlib import Path

# 新的system prompt（无作家名）
NEW_SYSTEM = (
    "你是一个专业的文学改写工具，仅执行文风转换任务。"
    "你的唯一职责是：对给定文本进行文风调整，使其呈现用词雅致古朴、情节委婉曲折、"
    "注重情感细节和人物心理的叙事风格，同时保持内容完全一致。"
    "严格禁止：(1)添加任何原文不存在的信息、情节、人物、对话；(2)删减或略去任何原文信息；"
    "(3)在改写结果后继续生成任何内容；(4)生成任何解释、评论、提示或元信息。"
    "输出格式：直接给出改写后的文本，不包含任何其他内容。改写完成后立即停止生成。"
)

# user prompt中需要替换的模式
USER_REPLACEMENTS = {
    "张恨水": "雅致细腻风格",
    "张恨水风格": "雅致细腻风格",
    "张恨水的笔法": "雅致细腻风格",
    "张恨水式": "雅致细腻风格",
    "民国章回体": "雅致细腻风格",
    "章回小说": "雅致细腻",
}


def clean_system_prompt(original: str) -> str:
    """清理system prompt，去掉作家名"""
    return NEW_SYSTEM


def clean_user_prompt(original: str) -> str:
    """清理user prompt，替换作家名为风格描述"""
    cleaned = original
    
    # 替换所有作家名相关表达
    for old_term, new_term in USER_REPLACEMENTS.items():
        cleaned = cleaned.replace(old_term, new_term)
    
    return cleaned


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/stage2_zhang_5000.jsonl", help="输入文件")
    parser.add_argument("--output", type=str, default="data/stage2_zhang_5000_cleaned.jsonl", help="输出文件")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"错误：输入文件不存在 {input_file}")
        return
    
    print(f"读取文件: {input_file}")
    
    modified_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                sample = json.loads(line)
                total_count += 1
                
                # 提取原始内容
                conversations = sample.get("conversations", [])
                if len(conversations) < 3:
                    # 格式不对，跳过
                    f_out.write(line)
                    continue
                
                original_system = conversations[0].get("content", "")
                original_user = conversations[1].get("content", "")
                assistant = conversations[2].get("content", "")
                
                # 清理prompt
                new_system = clean_system_prompt(original_system)
                new_user = clean_user_prompt(original_user)
                
                # 检查是否有修改
                if new_system != original_system or new_user != original_user:
                    modified_count += 1
                
                # 构建新样本
                new_sample = {
                    "conversations": [
                        {"role": "system", "content": new_system},
                        {"role": "user", "content": new_user},
                        {"role": "assistant", "content": assistant}
                    ]
                }
                
                # 保留record_id（如果有）
                if "_record_id" in sample:
                    new_sample["_record_id"] = sample["_record_id"]
                
                # 写入新文件
                f_out.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理样本时出错: {e}")
                # 保留原样
                f_out.write(line)
    
    print(f"\n✅ 处理完成！")
    print(f"   总样本数: {total_count}")
    print(f"   修改样本: {modified_count}")
    print(f"   输出文件: {output_file}")
    
    # 显示前3个样本的修改效果
    print(f"\n前3个样本的修改效果:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"\n样本{i+1}:")
            print(f"  System: {sample['conversations'][0]['content'][:120]}...")
            print(f"  User: {sample['conversations'][1]['content'][:120]}...")


if __name__ == "__main__":
    main()
