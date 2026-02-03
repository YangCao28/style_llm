"""转换 Stage2 数据集，去掉作家名字，改用通用风格描述"""

import json
from pathlib import Path

# 风格映射：作家名字 -> 风格描述
STYLE_MAPPING = {
    "倪匡": {
        "system_old": "你是倪匡式科幻作家",
        "system_new": "你是一位专业的文学风格转换助手。你擅长将文本改写为：语言简洁明快、逻辑严密、节奏紧凑、善于营造氛围的叙事风格。",
        "style_desc": "简洁紧凑风格"
    },
    "张恨水": {
        "system_old": "民国时期的章回小说家，擅长张恨水",
        "system_new": "你是一位专业的文学风格转换助手。你擅长将文本改写为：用词雅致古朴、情节委婉曲折、注重情感细节和人物心理的叙事风格。",
        "style_desc": "雅致细腻风格"
    },
}

def convert_prompt(text: str, author: str, style_desc: str) -> str:
    """转换user prompt，将作家名字替换为风格描述"""
    # 替换可能的模式
    patterns = [
        (f"{author}式", style_desc),
        (f"{author}的笔法", style_desc),
        (f"{author}小说的文风", style_desc),
        (f"{author}风格", style_desc),
        (author, style_desc),
    ]
    
    result = text
    for old, new in patterns:
        result = result.replace(old, new)
    
    return result

def main():
    input_file = Path("data/stage2_sample_5000.jsonl")
    output_file = Path("data/stage2_sample_5000_v2.jsonl")
    
    if not input_file.exists():
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line)
                conversations = data.get("conversations", [])
                
                if len(conversations) < 3:
                    f_out.write(line)
                    skipped_count += 1
                    continue
                
                # 获取 system 和 user 内容
                system_msg = conversations[0]
                user_msg = conversations[1]
                
                system_content = system_msg.get("content", "")
                user_content = user_msg.get("content", "")
                
                # 检测是哪个作家风格
                author_detected = None
                style_info = None
                
                for author, info in STYLE_MAPPING.items():
                    if author in system_content or author in user_content:
                        author_detected = author
                        style_info = info
                        break
                
                if author_detected and style_info:
                    # 替换 system prompt
                    system_msg["content"] = style_info["system_new"]
                    
                    # 替换 user prompt 中的作家名字
                    user_msg["content"] = convert_prompt(
                        user_content, 
                        author_detected, 
                        style_info["style_desc"]
                    )
                    
                    converted_count += 1
                else:
                    skipped_count += 1
                
                # 写入转换后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {e}")
                f_out.write(line)
                skipped_count += 1
    
    print(f"\n✅ 转换完成！")
    print(f"   转换: {converted_count} 条")
    print(f"   跳过: {skipped_count} 条")
    print(f"   输出: {output_file}")

if __name__ == "__main__":
    main()
