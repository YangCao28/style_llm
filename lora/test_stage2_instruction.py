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

DEFAULT_SYSTEM = "你是一个文学创作助手，擅长用各种风格改写文本。"
DEFAULT_USER = (
    "请以倪匡的文风改写这段话（原文不少于100字）：\n"
    "李长安站在桥头，看着江面上的雾气一点点被晨风撕开，河对岸稀疏的灯火像被水汽包裹的星星，时隐时现。\n"
    "桥下偶尔有早起的小货船缓慢驶过，发动机的轰鸣声被雾气闷得发闷，却在空旷的河面上来回回荡。\n"
    "他缩了缩脖子，把外套拉紧了一些，心里清楚，等雾彻底散尽，这座城里又会恢复白日里那种喧嚣，而他要做的事，也再没有退路。"
)


PRESET_CASES = {
    "ni_kuang": {
        "system": DEFAULT_SYSTEM,
        "user": DEFAULT_USER,
        "description": "倪匡式科幻 + 悬疑风格改写",
    },
    "zhang_henshui": {
        "system": "你是张恨水风格的小说家，擅长市井生活与情感描写。",
        "user": (
            "请用张恨水的文风改写下面这段（原文不少于100字）：\n"
            "她站在街角的小杂货铺门口，手里捏着找回来的几枚铜板，一时不知道该揣进袖子，还是继续翻来覆去地数。\n"
            "街巷尽头的天色已经被暮霭染成暗紫色，卖糖葫芦的吆喝声从巷口渐行渐远，只留下几串拖长的回音。\n"
            "对门茶馆里有人拍着桌子说笑，热气携着茶香从半掩的门缝里往外冒，她却只觉袖中那几枚硬币沉得很，像是拴着一桩说不清道不明的心事。"
        ),
        "description": "张恨水式市井情感风格改写",
    },
    "plain_to_style": {
        "system": DEFAULT_SYSTEM,
        "user": (
            "把下面这段朴素叙述改写得更有悬疑气氛（原文不少于100字）：\n"
            "今晚的风有点大，街道上几乎没有人，路边的树影被路灯拉得细长，在地上摇晃得像一层破碎的水纹。\n"
            "远处偶尔驶过一辆出租车，很快又被黑暗吞没，只留下车灯在转角处划出的一点亮痕。\n"
            "他站在小区门口，手机屏幕一次次亮起又熄灭，心里却越来越发毛——按理说，这个点本不该这么安静。"
        ),
        "description": "普通叙述 → 悬疑气氛加强",
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
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attn_impl",
        default="sdpa",
        help="Attention impl for inference (sdpa, eager, or flash_attention_2).",
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
    print(f"Loading Stage-2 model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
