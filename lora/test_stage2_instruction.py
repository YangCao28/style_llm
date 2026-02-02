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
)

ORIGINAL_TEXT = (
    "朱元璋坐于殿中，烛影摇红，面带阴鸷之色，左右侍立，皆为陈东等文吏。"
    "“暗衣卫指挥使，”朱元璋言，“持此腰牌，监察百僚，先斩后奏。”陈东接了腰牌，心中甚喜。"
    "又命将一份文书交出。那纸上月光之下，隐隐有字迹。“剥皮实草令，”朱元璋曰：“贪墨通敌者，剥皮填草，悬头示众。速宜办理。”"
    "陈东领旨，恭敬地将腰牌文书收好，退出殿外。殿内复寂，惟见烛火微明，忽有一声夜枭悲啸，自远方来，接着东南角上火光一闪，照得夜空一片。"
    "朱元璋便至窗边，望那火处，指尖微颤。火光跳了一会，渐渐低矮，最后被黑夜吞噬。远远的地方又有呼喊之声，渐渐止息。他料想陈东已行事。"
    "他仍坐回案前，取过一卷奏章，却未启视。殿内惟闻其呼吸之声。约莫两刻之后，脚步之声渐近。"
    "陈东疾趋而入，衣襟犹湿，手中捧着一个乌木盒儿。“陛下，秦桧府中已肃清，内外毕静。”"
    "陈东语声微促，尚带风尘劳顿。“通敌密信在此，证据确凿。”朱元璋接了乌木盒儿，手抚雕纹，揭开铜扣，打开盖子，里面整齐叠着十二封密函。"
    "此信乃特制药水所写，隐时无痕，现时字迹悉出，墨色如新。他抽出一封，展视。"
    "那封书信上款是金国元帅完颜昌，落款为秦桧花押。字里行间，尽是军机民情、割地赔款之事，历历如账簿。"
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
    print(f"model_name_or_path = {args.model_name_or_path}")

    # 优先当作本地目录使用；如果目录不存在，再回退为 HF 仓库名
    if model_path.exists():
        print(f"Loading Stage-2 model from local folder: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
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
