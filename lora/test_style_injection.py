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

DEFAULT_PROMPT = """只见龙舟停于扬州江面，船舱中灯火摇曳，照见金册之封。金箔在光中晃晃，几个文官围绕，大气不敢出一口。朱元璋已知，乃完颜昌所献和议金册也。遂接来，揭开。称臣，纳贡，割地，淮河以北尽数让出，每年银绢各三十万。忽有一宦官伏地而进，声音颤颤，言："官家三思...金兵势大，南渡方可保宗庙啊！"朱元璋不语，只手执金册，两手扯碎。嘈嘈切切，金箔纸页从中撕裂。又扯了几下，碎屑纷扬。那宦官尚欲谏止："往杭州，暂且避锋"，朱元璋瞥了他一眼，抽出佩剑。剑光一闪，宦官身首异处，无言可答。舱内寂然，惟有江水拍击船底之声。他收剑，至案前，铺黄绫，执朱笔。"念。"他一面写，一面说。李纲趋至，接了，朗声读出："返航临安，停止南逃。即日，抗金诸事，由李纲全权处置，总军政。"李纲背脊挺立，声响于舱内。朱元璋盖上玉玺，将圣旨交付与李纲："去做。"李纲受命，正色而出。朱元璋至船头，江风吹拂，衣裾飘扬，岸上隐隐有人攒聚，未散者众。他开口，语虽不甚高，却一字一句，自风中传入："吾为朱元璋，亦是赵构。金人欲灭吾国，吾不去矣。"他顿了顿，向北望去："自此，不讲和议，惟死战而已。"吩咐毕，龙舟徐行，船头破浪，众船跟随，桨橹齐动，破浪前行。"""


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
