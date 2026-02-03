"""Quick smoke test for Stage-2 instruction-tuned checkpoints.

Loads a Stage-2 checkpoint (which already includes the Stage-1 LoRA),
feeds a system+user conversation, and prints the assistant reply so you
can visually inspect whether the instruction style and literary style
look correct.

Usage examples:

    # Test the main Stage-2 run
    python -m lora.test_stage2_instruction \
        --model_name_or_path stage2_instruction_tuning

    # Or test a specific checkpoint
    python -m lora.test_stage2_instruction \
        --model_name_or_path stage2_instruction_tuning/checkpoint-158

    # Or test the alpha-enhanced run
    python -m lora.test_stage2_instruction \
        --model_name_or_path stage2_instruction_alpha
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM = (
    "你是一个文学改写助手，只负责调整文风和措辞。"
    "改写时必须完整保留原文提供的全部信息，不得扩写情节，也不得删减内容。"
    "你的回复必须在完成改写后立即结束，不得继续生成任何对话、提示或新任务。"
)

ORIGINAL_TEXT = (
    "严冬时节，鹅毛一样的大雪片在天空中到处飞舞着，有一个王后坐在王宫里的一扇窗子边，正在为她的女儿做针线活儿，"
    "寒风卷着雪片飘进了窗子，乌木窗台上飘落了不少雪花。她抬头向窗外望去，一不留神，针刺进了她的手指，"
    "红红的鲜血从针口流了出来，有三点血滴落在飘进窗子的雪花上。"
)

DEFAULT_USER = (
    "请在不增删任何信息的前提下，用更紧张、悬疑的文风改写下面这段：\n" + ORIGINAL_TEXT
)


PRESET_CASES = {
    "ni_kuang": {
        "system": DEFAULT_SYSTEM,
        "user": DEFAULT_USER,
        "description": "悬疑风格改写测试（不增删信息）",
    },
    "zhang_henshui": {
        "system": (
            "你擅长描写市井生活与细腻情感。"
            "在改写时只允许改变用词和语气，不得新增桥段或删减信息。"
        ),
        "user": (
            "请在不增删任何信息的前提下，把下面这段改写成更有市井情调、细腻情感的文风：\n" + ORIGINAL_TEXT
        ),
        "description": "市井情感风格改写测试（不增删信息）",
    },
    "plain_to_style": {
        "system": DEFAULT_SYSTEM,
        "user": (
            "请在不增删任何信息的前提下，把下面这段朴素叙述改写得更有悬疑气氛：\n" + ORIGINAL_TEXT
        ),
        "description": "悬疑气氛加强测试（不增删信息）",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Stage-2 instruction-tuned checkpoint")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the Stage-2 checkpoint folder (or HF repo id).",
    )
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM, help="System prompt.")
    parser.add_argument("--user", type=str, default=DEFAULT_USER, help="User message.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESET_CASES.keys()),
        help="Use a built-in test case (overrides --system/--user if set).",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=None,
        help="Optional file containing multiple user prompts. One JSONL per line with 'system'/'user', or plain text (one prompt per line).",
    )
    # 为了支持至少 ~100 字的输出，默认给得稍微长一点
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attn_impl",
        default="sdpa",
        help="Attention impl for inference (sdpa, eager, or flash_attention_2).",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name for loading tokenizer if checkpoint doesn't have it (e.g., Qwen/Qwen2.5-8B-Base).",
    )
    return parser.parse_args()


def build_chat_prompt(system: str, user: str) -> str:
    """Build a single-turn chat prompt using the same format as training.

    Training used messages like:
      <|im_start|>system\n...<|im_end|>\n
      <|im_start|>user\n...<|im_end|>\n
      <|im_start|>assistant\n
    Here we stop before closing the assistant block so generation continues it.
    """

    parts = [
        f"<|im_start|>system\n{system}<|im_end|>",
        f"<|im_start|>user\n{user}<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_path = Path(args.model_name_or_path)
    print(f"model_name_or_path = {args.model_name_or_path}")

    # 优先当作本地目录使用；如果目录不存在，再回退为 HF 仓库名
    if model_path.exists():
        print(f"Loading Stage-2 model from local folder: {model_path}")
        
        # 尝试从 checkpoint 加载 tokenizer，如果失败则从基础模型加载
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("✓ Tokenizer loaded from checkpoint")
        except (OSError, ValueError, ImportError) as e:
            print(f"⚠ Checkpoint 中没有 tokenizer，尝试从基础模型加载...")
            
            # 尝试多种方式找到基础模型
            base_model_path = None
            
            # 1. 使用命令行参数
            if args.base_model_name:
                base_model_path = args.base_model_name
                print(f"  使用命令行参数: {base_model_path}")
            
            # 2. 尝试从 config.json 读取 _name_or_path
            if not base_model_path:
                config_path = model_path / "config.json"
                if config_path.exists():
                    import json
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        base_model_path = config.get("_name_or_path")
                        if base_model_path:
                            print(f"  从 config.json 读取: {base_model_path}")
            
            # 3. 尝试从父目录或祖父目录找 Stage1 模型
            if not base_model_path:
                # stage2_instruction_tuning_corrected/checkpoint-16 -> stage1_style_injection
                parent_dir = model_path.parent.parent
                stage1_path = parent_dir / "stage1_style_injection"
                if stage1_path.exists() and (stage1_path / "config.json").exists():
                    with open(stage1_path / "config.json", "r") as f:
                        config = json.load(f)
                        base_model_path = config.get("_name_or_path")
                        if base_model_path:
                            print(f"  从 Stage1 config 读取: {base_model_path}")
            
            # 4. 使用常见的模型名称作为回退
            if not base_model_path:
                possible_models = [
                    "Qwen/Qwen2.5-8B-Base",
                    "Qwen/Qwen2.5-7B-Base",
                    "Qwen/Qwen2-7B-Base",
                ]
                print("  尝试常见模型名称...")
                for model_name in possible_models:
                    print(f"    尝试: {model_name}")
                    base_model_path = model_name
                    break
            
            if not base_model_path:
                raise ValueError(
                    "无法确定基础模型名称。请使用 --base_model_name 参数指定，"
                    "例如: --base_model_name Qwen/Qwen2.5-8B-Base"
                )
            
            print(f"  Loading tokenizer from: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        model_load_id = model_path
    else:
        print(f"⚠ 本地找不到目录: {model_path}，将尝试作为 Hugging Face 模型仓库加载。")
        model_load_id = args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_load_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_load_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_impl,
    )
    model.eval()
    # 根据 preset 或手动 system/user 构造一个或多个测试用例
    test_cases = []

    if args.input_file is not None:
        # 文件支持两种格式：
        # 1) JSONL，每行形如 {"system": "...", "user": "..."}
        # 2) 纯文本，每行作为 user，system 使用默认或 preset 的 system
        print(f"Loading prompts from: {args.input_file}")
        with args.input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    system = rec.get("system") or args.system
                    user = rec.get("user") or ""
                except json.JSONDecodeError:
                    # 当作纯文本 user
                    system = args.system
                    user = line
                test_cases.append((system, user))
    elif args.preset:
        preset = PRESET_CASES[args.preset]
        print(f"Using preset='{args.preset}': {preset['description']}")
        test_cases.append((preset["system"], preset["user"]))
    else:
        test_cases.append((args.system, args.user))

    for idx, (system, user) in enumerate(test_cases, start=1):
        prompt = build_chat_prompt(system, user)
        print("\n" + "=" * 80)
        print(f"Test case #{idx}")
        print("----- System -----")
        print(system)
        print("----- User -----")
        print(user)
        print("----- Raw Prompt (truncated) -----")
        print(prompt[:400] + ("..." if len(prompt) > 400 else ""))
        print("=" * 80)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                # Stop tokens to prevent unwanted continuation
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )

        completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("===== Assistant Reply =====")
        # 简单做个切分：去掉提示部分，只看 assistant 段
        if prompt in completion:
            print(completion[len(prompt) :])
        else:
            print(completion)


if __name__ == "__main__":
    main()
