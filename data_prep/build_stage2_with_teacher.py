"""使用教师模型从 Stage 1 语料生成 Stage 2 指令微调数据

流程：
1. 从 combined_dataset_uniform.jsonl 中按作者/风格筛选高质量片段
2. 使用教师模型（如 Qwen/Llama）将原文总结为白话文
3. 生成指令格式的对话数据（system + user + assistant）
4. 按比例采样：倪匡40%、张恨水30%、地煞15%、梁羽生15%
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from tqdm import tqdm
import requests


# 风格定义和系统提示词
STYLE_CONFIG = {
    "倪匡": {
        "system": "你是倪匡式科幻作家，擅长将科学幻想与悬疑推理结合，文笔简洁明快，逻辑严密，善于制造悬念。",
        "keywords": ["科幻", "悬疑", "卫斯理", "推理"],
        "dataset_pattern": r"ni_kuang|倪匡|wesley",
    },
    "张恨水": {
        "system": "你是民国时期的章回小说家，擅长张恨水、鸳鸯蝴蝶派的叙事风格。用词雅致，情节曲折，擅长描写世情人心。",
        "keywords": ["民国", "章回体", "言情", "世情"],
        "dataset_pattern": r"zhang.*henshui|张恨水",
    },
    "梁羽生": {
        "system": "你是武侠小说大师，精通梁羽生的笔法。文风儒雅，诗词歌赋信手拈来，注重历史背景和文化底蕴。",
        "keywords": ["武侠", "梁羽生", "江湖", "诗词"],
        "dataset_pattern": r"liang.*yusheng|梁羽生|wuxia",
    },
    "现代玄幻": {
        "system": "你是古典文学专家，擅长用文言半白的笔法，模仿明清小说的叙事风格。",
        "keywords": ["古典", "文言", "传统"],
        "dataset_pattern": r"disha",
    },
}

# 用户提示词模板
USER_PROMPT_TEMPLATES = [
    "将这段白话文改写为{style}风格：\n{text}",
    "用{author}的笔法重写这段话：\n{text}",
    "请以{style}小说的文风改写：\n{text}",
    "把下面的文字转换为{style}：\n{text}",
    "模仿{author}改写这段内容：\n{text}",
]

# 过滤规则
FILTER_PATTERNS = [
    re.compile(r"第[一二三四五六七八九十百千0-9]+[章回节卷]", re.IGNORECASE),
    re.compile(r"目\s*录", re.IGNORECASE),
    re.compile(r"(前|后)?言", re.IGNORECASE),
    re.compile(r"作者[:：]", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"(八二|82|8\.2)小说网", re.IGNORECASE),
    re.compile(r"(简|繁)体.*阅读", re.IGNORECASE),
    re.compile(r"无弹窗", re.IGNORECASE),
    re.compile(r"下一章|上一章|返回目录", re.IGNORECASE),
    re.compile(r"且听下回分解", re.IGNORECASE),
    re.compile(r"欲知.*如何|欲知后事", re.IGNORECASE),
    re.compile(r"手机端.*阅读", re.IGNORECASE),
    re.compile(r"&\w+;", re.IGNORECASE),  # HTML entities
]


@dataclass
class TeacherModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_input_length: int = 1024
    max_output_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_api: bool = False
    api_config_path: Optional[str] = None


def load_api_config(config_path: str) -> Dict:
    """加载 API 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def simplify_with_api(
    styled_text: str,
    api_config: Dict,
    max_length: int = 1024
) -> Optional[str]:
    """使用 DeepSeek API 将风格化文本简化为白话文"""
    
    if len(styled_text) > max_length:
        styled_text = styled_text[:max_length]
    
    system_prompt = api_config.get("system_prompt", "请将下面的文学作品改写为简洁的现代白话文。")
    
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": styled_text}
                ],
                "temperature": api_config["generation"]["temperature"],
                "max_tokens": api_config["generation"]["max_tokens"]
            },
            timeout=api_config["generation"]["timeout"]
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"API 错误 {response.status_code}: {response.text}")
            return None
    
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None


def simplify_with_teacher_model(
    styled_text: str,
    model,
    tokenizer,
    config: TeacherModelConfig,
    api_config: Optional[Dict] = None
) -> Optional[str]:
    """使用教师模型将风格化文本简化为白话文"""
    
    # 如果使用 API
    if config.use_api and api_config:
        return simplify_with_api(styled_text, api_config, config.max_input_length)
    
    # 本地模型
    if model is None or tokenizer is None:
        return None
    
    import torch
    
    # 截断过长文本
    if len(styled_text) > config.max_input_length:
        styled_text = styled_text[:config.max_input_length]
    
    prompt = f"""请将下面的文学作品改写为简洁的现代白话文。要求：
1. 保持内容完整，不删减任何情节和对话
2. 去除所有广告、网站链接、章节标题等无关内容
3. 将文言、古雅表达改为通俗易懂的现代汉语
4. 简化复杂句式，但不省略关键信息

原文：
{styled_text}

白话文改写："""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_output_length,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return result.strip()
    
    except Exception as e:
        print(f"生成失败: {e}")
        return None


def load_teacher_model(config: TeacherModelConfig):
    """加载教师模型用于生成白话文摘要"""
    if config.use_api:
        print(f"将使用 DeepSeek API: {config.api_config_path}")
        api_config = load_api_config(config.api_config_path)
        return None, None, api_config
    
    print(f"正在加载本地教师模型: {config.model_name}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()
        print("✓ 教师模型加载完成")
        return model, tokenizer, None
    except ImportError:
        print("⚠️  未安装 torch/transformers，请使用 --use_api")
        return None, None, None


def is_valid_text(text: str) -> bool:
    """检查文本是否包含不需要的内容"""
    if len(text.strip()) < 100:  # 太短
        return False
    
    for pattern in FILTER_PATTERNS:
        if pattern.search(text):
            return False
    
    # 检查是否有足够的汉字
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < len(text) * 0.7:  # 汉字占比低于70%
        return False
    
    return True


def simplify_with_teacher_model(
    styled_text: str,
    model,
    tokenizer,
    config: TeacherModelConfig
) -> Optional[str]:
    """使用教师模型将风格化文本简化为白话文"""
    
    # 截断过长文本
    if len(styled_text) > config.max_input_length:
        styled_text = styled_text[:config.max_input_length]
    
    prompt = f"""请将下面的文学作品片段改写为简洁的现代白话文，要求：
1. 保留核心情节和人物动作
2. 去除华丽辞藻，使用日常用语
3. 简化长句，使其通俗易懂
4. 长度控制在原文的50-70%

原文：
{styled_text}

白话文改写："""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_output_length,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return result.strip()
    
    except Exception as e:
        print(f"生成失败: {e}")
        return None


def classify_dataset(record: Dict) -> Optional[str]:
    """根据记录判断属于哪个风格"""
    dataset = record.get("dataset", "")
    source_file = record.get("source_file", "")
    text = record.get("text", "")
    
    combined_text = f"{dataset} {source_file}".lower()
    
    for style, config in STYLE_CONFIG.items():
        if re.search(config["dataset_pattern"], combined_text, re.IGNORECASE):
            return style
    
    return None


def load_and_filter_data(input_path: Path) -> Dict[str, List[Dict]]:
    """加载并按风格分类数据"""
    print(f"正在加载数据: {input_path}")
    
    style_data = {style: [] for style in STYLE_CONFIG.keys()}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="筛选数据"):
            try:
                record = json.loads(line)
                text = record.get("text", "")
                
                if not is_valid_text(text):
                    continue
                
                style = classify_dataset(record)
                if style:
                    style_data[style].append(record)
            
            except json.JSONDecodeError:
                continue
    
    for style, records in style_data.items():
        print(f"  {style}: {len(records)} 条")
    
    return style_data


def generate_stage2_sample(
    styled_text: str,
    plain_text: str,
    style: str
) -> Dict:
    """生成 Stage 2 训练样本"""
    
    config = STYLE_CONFIG[style]
    system_prompt = config["system"]
    
    # 随机选择用户提示词模板
    user_template = random.choice(USER_PROMPT_TEMPLATES)
    user_prompt = user_template.format(
        style=style,
        author=random.choice(config["keywords"]),
        text=plain_text
    )
    
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": styled_text}
        ],
        "metadata": {
            "style": style,
            "plain_length": len(plain_text),
            "styled_length": len(styled_text),
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_api", action="store_true", help="使用 DeepSeek API 而非本地模型")
    parser.add_argument("--api_config", type=Path, default=Path("data_prep/llm_config.json"), help="API 配置文件路径")
    parser.add_argument("--input", type=Path, default=Path("data/dataset/combined_dataset_uniform.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/stage2_sft_6k.json"))
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_samples", type=int, default=6000)
    parser.add_argument("--skip_teacher", action="store_true", help="跳过教师模型，使用规则简化")
    args = parser.parse_args()
    
    # 目标分布
    distribution = {
        "倪匡": int(args.target_samples * 0.40),      # 2400
        "张恨水": int(args.target_samples * 0.30),     # 1800
        "梁羽生": int(args.target_samples * 0.15),     # 900
        "现代玄幻": int(args.target_samples * 0.15),   # 900
    }
    
    print(f"目标样本分布:")
    for style, count in distribution.items():
        print(f"  {style}: {count} 条")
    
    # 加载数据
    style_data = load_and_filter_data(args.input)
    
    api_config = None
    if not args.skip_teacher:
        teacher_config = TeacherModelConfig(
            model_name=args.teacher_model,
            use_api=args.use_api,
            api_config_path=str(args.api_config) if args.use_api else None
        )
        teacher_model, teacher_tokenizer, api_config])
        if available < target_count:
            print(f"⚠️  {style} 数据不足: 需要 {target_count}，实际 {available}")
    
    # 加载教师模型
    if not args.skip_teacher:
        teacher_config = TeacherModelConfig(model_name=args.teacher_model)
        teacher_model, teacher_tokenizer = load_teacher_model(teacher_config)
    
    # 生成训练数据
    stage2_samples = []
    
    for style, target_count in distribution.items():
        print(f"\n正在生成 {style} 样本...")
        available_data = style_data[style]
        
        if len(available_data) < target_count:
            print(f"  数据不足，将生成 {len(available_data)} 条")
            sampled = available_data
        else:
            sampled = random.sample(available_data, target_count)
        
        for record in tqdm(sampled, desc=f"生成{style}"):
            styled_text = record[",
                    api_configtext"]
            
            if args.skip_teacher:
                # 简单规则：每3句取1句
                sentences = re.split(r'[。！？]', styled_text)
                plain_text = "。".join(sentences[::3])[:300]
            else:
                # 使用教师模型
                plain_text = simplify_with_teacher_model(
                    styled_text,
                    teacher_model,
                    teacher_tokenizer,
                    teacher_config
                )
                
                if not plain_text:
                    continue
            
            sample = generate_stage2_sample(styled_text, plain_text, style)
            stage2_samples.append(sample)
    
    # 打乱顺序
    random.shuffle(stage2_samples)
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(stage2_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已生成 {len(stage2_samples)} 条 Stage 2 训练样本")
    print(f"✓ 保存到: {args.output}")
    
    # 统计
    style_counts = {}
    for sample in stage2_samples:
        style = sample["metadata"]["style"]
        style_counts[style] = style_counts.get(style, 0) + 1
    
    print(f"\n实际生成分布:")
    for style, count in style_counts.items():
        print(f"  {style}: {count} 条")


if __name__ == "__main__":
    main()
