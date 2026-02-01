"""Quick smoke test for Stage-1 LoRA checkpoints.

Loads the base model plus LoRA adapter, feeds a prompt, and prints the
continuation so you can visually inspect whether the style injection took effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "今日天气阴沉，"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Stage-1 style-injection LoRA")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path or HF repo id.")
    parser.add_argument(
        "--lora_path",
        type=Path,
        required=True,
        help="Directory containing the LoRA adapter (stage1_style_injection output).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn_impl", default="flash_attention_2", help="Attention impl for inference (default FA2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_impl,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

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
    print("===== Prompt =====")
    print(args.prompt)
    print("===== Completion =====")
    # strip the prompt to show only continuation for clarity
    print(completion[len(args.prompt):])


if __name__ == "__main__":
    main()
