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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM = "你是一个文学创作助手，擅长用各种风格改写文本。"
DEFAULT_USER = "请以倪匡的文风改写这段话：\n李长安站在桥头，看着江面上的雾气一点点散去。"


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
    parser.add_argument("--max_new_tokens", type=int, default=256)
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

    prompt = build_chat_prompt(args.system, args.user)
    print("===== Prompt (truncated) =====")
    print(prompt)
    print("==============================")

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
