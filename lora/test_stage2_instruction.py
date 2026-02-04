"""Quick smoke test for Stage-2 instruction-tuned checkpoints.

Three testing modes (determined by provided arguments):

1. Base Model Mode:
   --base_model <path>

2. Single LoRA Mode:
   --lora_model <path>

3. Dual LoRA Mode (Stacked Adapters):
   --style_adapter <path> --instruct_adapter <path>

Usage examples:

    # Test base model
    python -m lora.test_stage2_instruction --base_model Qwen3-8B-Base

    # Test Stage1 (style only)
    python -m lora.test_stage2_instruction --lora_model stage1_style_injection/checkpoint-531

    # Test Stage2 (style + instruct stacked)
    python -m lora.test_stage2_instruction \
        --style_adapter stage1_style_injection/checkpoint-531 \
        --instruct_adapter stage2_instruct_new_adapter

    # Override base model detection
    python -m lora.test_stage2_instruction \
        --base_model /path/to/Qwen3-8B-Base \
        --lora_model stage1_style_injection/checkpoint-531
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_SYSTEM = "你是一名优雅的文学改写作家，擅长把现代白话润色成更讲究、更有韵味的华美文本。要求：\n1. 严格保留原文的事实与情节，不新增信息。\n2. 用更讲究的词汇、句式和古雅的表达，让语言更有气韵，但保持可读性。\n3. 输出保持为中文段落，不要添加任何解释。"

ORIGINAL_TEXT = "朱元璋坐于殿中，烛影摇红，面带阴鸷之色，左右侍立，皆为陈东等文吏。\"暗衣卫指挥使，\"朱元璋言，\"持此腰牌，监察百僚，先斩后奏。\"陈东接了腰牌，心中甚喜。又命将一份文书交出。那纸上月光之下，隐隐有字迹。\"剥皮实草令，\"朱元璋曰：\"贪墨通敌者，剥皮填草，悬头示众。速宜办理。\"陈东领旨，恭敬地将腰牌文书收好，退出殿外。殿内复寂，惟见烛火微明，忽有一声夜枭悲啸，自远方来，接着东南角上火光一闪，照得夜空一片。朱元璋便至窗边，望那火处，指尖微颤。火光跳了一会，渐渐低矮，最后被黑夜吞噬。远远的地方又有呼喊之声，渐渐止息。他料想陈东已行事。他仍坐回案前，取过一卷奏章，却未启视。殿内惟闻其呼吸之声。约莫两刻之后，脚步之声渐近。陈东疾趋而入，衣襟犹湿，手中捧着一个乌木盒儿。\"陛下，秦桧府中已肃清，内外毕静。\"陈东语声微促，尚带风尘劳顿。\"通敌密信在此，证据确凿。\"朱元璋接了乌木盒儿，手抚雕纹，揭开铜扣，打开盖子，里面整齐叠着十二封密函。此信乃特制药水所写，隐时无痕，现时字迹悉出，墨色如新。他抽出一封，展视。那封书信上款是金国元帅完颜昌，落款为秦桧花押。字里行间，尽是军机民情、割地赔款之事，历历如账簿。他逐张看过，不慌不忙。殿内只听得纸页翻动之声。烛影摇红，映在身后墙上。一面看，一面将信纸折叠起来，放回盒内，盖好。\"万俟卨、沈该亦系府中同党，并已拿获。\"陈东禀报说毕，又道，\"吴才人率女锦衣卫封锁内院，无人逃出。\"朱元璋点头道：\"知道了。\"一面摩挲木匣，一面说道：\"人犯押入诏狱，严加看守。此件密信，抄录副本，原件封存。\"他又停了一会儿，抬头道：\"三日之内，午门外，当众宣判。\"陈东领命退出。朱元璋独自坐着，用手指在木匣上轻轻划着，想道：\"秦桧通敌，证据确凿。\"便唤赵鼎、李纲进来。二人显然已起身等候，衣冠整齐。朱元璋遂将秦桧之事细说一遍，二人面面相觑，神色沉重。赵鼎道：\"陛下，若行公审，震动不小。\"李纲接口道：\"臣等已调兵维护临安各门要道，防余党作乱。\"朱元璋道：\"拟旨，布告天下。\"又向赵鼎道：\"罪名、证据、刑律，都要一一列明。\"赵鼎应诺，遂退，脚步声渐远，终不可闻。"

DEFAULT_USER = (
    "请将以下现代白话润色成雅致的文学文本：\n\n"
    + ORIGINAL_TEXT
)


PRESET_CASES = {
    "elegant_style": {
        "system": DEFAULT_SYSTEM,
        "user": DEFAULT_USER,
        "description": "雅致细腻风格改写（与训练数据格式完全一致）",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Stage-2 instruction-tuned checkpoint")
    
    # 模型路径参数（根据提供的参数自动判断模式）
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path (e.g., Qwen3-8B-Base). If only this is provided, test pure base model.",
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="Single LoRA adapter path (e.g., stage1_style_injection/checkpoint-531)",
    )
    parser.add_argument(
        "--style_adapter",
        type=str,
        default=None,
        help="Style adapter path (e.g., stage1_style_injection/checkpoint-531). Use with --instruct_adapter for dual mode.",
    )
    parser.add_argument(
        "--instruct_adapter",
        type=str,
        default=None,
        help="Instruct adapter path (e.g., stage2_instruct_new_adapter). Use with --style_adapter for dual mode.",
    )
    
    # 测试参数
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
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
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
    parts = []
    parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 🔑 根据提供的参数自动判断模式
    # 判断逻辑：
    # 1. style_adapter + instruct_adapter -> 双adapter模式
    # 2. lora_model -> 单adapter模式
    # 3. base_model (且无其他adapter) -> 基座模式
    # 4. 否则 -> 参数错误
    
    if args.style_adapter and args.instruct_adapter:
        # 双 LoRA 模式：分别加载 style 和 instruct adapters
        print(f"Mode: Dual LoRA (Stacked Adapters)\n")
        print(f"Style adapter:    {args.style_adapter}")
        print(f"Instruct adapter: {args.instruct_adapter}")
        
        # 获取 base model（优先级：--base_model > adapter config > 当前目录）
        base_model_name = args.base_model
        if not base_model_name:
            style_config_path = Path(args.style_adapter) / "adapter_config.json"
            if style_config_path.exists():
                try:
                    with open(style_config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                    if base_model_name:
                        print(f"Base: {base_model_name} (from style adapter config)")
                except Exception as e:
                    print(f"⚠️  Failed to read style adapter config: {e}")
        
        if not base_model_name:
            default_base = Path("Qwen3-8B-Base")
            if default_base.exists():
                base_model_name = str(default_base)
                print(f"Base: {base_model_name} (auto-detected in current dir)")
            else:
                raise ValueError(
                    "❌ Cannot determine base model.\n"
                    "Use --base_model Qwen3-8B-Base"
                )
        else:
            if args.base_model:
                print(f"Base: {base_model_name}")
        # 加载 tokenizer 和 base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )
        
        # 🔑 加载第一个 adapter (style)
        model = PeftModel.from_pretrained(
            base_model,
            args.style_adapter,
            adapter_name="style",
            torch_dtype=torch.bfloat16
        )
        print(f"✓ Loaded style adapter")
        
        # 🔑 加载第二个 adapter (instruct)
        model.load_adapter(args.instruct_adapter, adapter_name="instruct")
        print(f"✓ Loaded instruct adapter")
        
        # 显示叠加信息
        adapters = list(model.peft_config.keys())
        print(f"\n🔗 Stacking adapters:")
        for adapter_name in adapters:
            print(f"  ✓ {adapter_name}")
        print(f"\n✓ All adapters will be stacked during inference")
        print(f"  Formula: W = W_base + ΔW_style + ΔW_instruct")
    
    elif args.lora_model:
        # 单 LoRA 模式
        print(f"Mode: Single LoRA\n")
        print(f"LoRA: {args.lora_model}")
        
        # 优先顺序：--base_model > adapter_config.json > 当前目录
        base_model_name = args.base_model
        
        if not base_model_name:
            # 从 adapter_config.json 读取
            adapter_config_path = Path(args.lora_model) / "adapter_config.json"
            if adapter_config_path.exists():
                try:
                    with open(adapter_config_path, "r", encoding="utf-8") as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                    if base_model_name:
                        print(f"Base: {base_model_name} (from adapter_config.json)")
                except Exception as e:
                    print(f"⚠️  Failed to read adapter_config.json: {e}")
        
        if not base_model_name:
            default_base = Path("Qwen3-8B-Base")
            if default_base.exists():
                base_model_name = str(default_base)
                print(f"Base: {base_model_name} (auto-detected in current dir)")
            else:
                raise ValueError(
                    "❌ Cannot determine base model.\n"
                    "Solutions:\n"
                    "  1. Use --base_model Qwen3-8B-Base\n"
                    "  2. Ensure Qwen3-8B-Base exists in current directory\n"
                    "  3. Make sure adapter_config.json contains base_model_name_or_path"
                )
        else:
            if args.base_model:
                print(f"Base: {base_model_name}")
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        
        # 加载 base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )
        
        # 加载 LoRA adapter
        model = PeftModel.from_pretrained(base_model, str(args.lora_model), torch_dtype=torch.bfloat16)
        
        # 检查加载的 adapters
        if hasattr(model, 'peft_config'):
            adapters = list(model.peft_config.keys())
            if adapters:
                print(f"Adapter: {adapters[0]}")
                print(f"✓ Single adapter mode")
            else:
                print("⚠️  No adapters found in peft_config")
        else:
            print("⚠️  Model does not have peft_config attribute")
    
    elif args.base_model:
        # 基座模型模式
        print(f"Mode: Base Model\n")
        print(f"Model: {args.base_model}")
        
        base_model_name = args.base_model
        
        # 尝试当前文件夹
        model_path = Path(base_model_name)
        if not model_path.exists():
            local_base = Path("Qwen3-8B-Base")
            if local_base.exists():
                base_model_name = str(local_base)
                print(f"(Using: {base_model_name})")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )
    
    else:
        # 参数错误
        raise ValueError(
            "❌ Invalid arguments. Must provide one of:\n"
            "  1. --base_model <path>                          (test base model)\n"
            "  2. --lora_model <path>                          (test single adapter)\n"
            "  3. --style_adapter <path> --instruct_adapter <path>  (test dual adapters)\n"
            "\nExamples:\n"
            "  python -m lora.test_stage2_instruction --base_model Qwen3-8B-Base\n"
            "  python -m lora.test_stage2_instruction --lora_model stage1_style_injection/checkpoint-531\n"
            "  python -m lora.test_stage2_instruction --style_adapter stage1_style_injection/checkpoint-531 --instruct_adapter stage2_instruct_new_adapter"
        )
        
        # 加载 LoRA adapters（会自动加载该目录下的所有 adapters）
        model = PeftModel.from_pretrained(base_model, str(model_path), torch_dtype=torch.bfloat16)
        
        # 🔑 检查加载的 adapters
        if hasattr(model, 'peft_config'):
            adapters = list(model.peft_config.keys())
            if adapters:
                print(f"Adapters: {adapters}")
                
                # 如果有多个 adapters，说明是 Stage2（style + instruct 叠加）
                if len(adapters) > 1:
                    print(f"\n🔗 Stacking adapters:")
                    for adapter_name in adapters:
                        print(f"  ✓ {adapter_name}")
                    
                    # PEFT 默认行为：所有 adapters 自动叠加（相加）
                    # W_final = W_base + ΔW_adapter1 + ΔW_adapter2 + ...
                    print(f"\n✓ All adapters will be stacked during inference")
                    print(f"  Formula: W = W_base + ΔW_{adapters[0]}" + 
                          "".join(f" + ΔW_{a}" for a in adapters[1:]))
                else:
                    print(f"✓ Single adapter mode")
            else:
                print("⚠️  No adapters found in peft_config")
        else:
            print("⚠️  Model does not have peft_config attribute")
        # 尝试当前文件夹
        if not model_path.exists():
            local_base = Path("Qwen3-8B-Base")
            if local_base.exists():
                model_path = local_base
                print(f"(Using: {model_path})")
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"\n✓ Ready\n")
    
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
        print("=" * 80)
        print(f"Test #{idx}")
        print("=" * 80 + "\n")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 配置停止 tokens
        stop_token_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        if im_end_id and im_end_id != tokenizer.unk_token_id:
            stop_token_ids.append(im_end_id)
        if im_start_id and im_start_id != tokenizer.unk_token_id:
            stop_token_ids.append(im_start_id)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        completion = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        
        # 提取 assistant 回复
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in completion:
            pos = completion.rfind(assistant_marker)
            reply = completion[pos + len(assistant_marker):]
            # 移除结束标记
            if reply.endswith("<|im_end|>"):
                reply = reply[:-len("<|im_end|>")]
            elif "<|im_end|>" in reply:
                reply = reply[:reply.rfind("<|im_end|>")]
            # 移除 <|endoftext|> 标记
            if "<|endoftext|>" in reply:
                reply = reply[:reply.find("<|endoftext|>")]
        else:
            reply = completion
        
        # 清理多余的前缀内容（模型可能生成的礼貌用语和代码块标记）
        reply = reply.strip()
        
        # 移除常见的多余前缀
        unwanted_prefixes = [
            "好的，请稍候片刻。",
            "好的，",
            "```python",
            "```",
            "Assistant:",
            "assistant:",
        ]
        
        for prefix in unwanted_prefixes:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()
        
        # 如果有多行，移除空行和只包含代码块标记的行
        lines = reply.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in ['```', '```python', 'Assistant:', 'assistant:']:
                cleaned_lines.append(line)
        
        reply = '\n'.join(cleaned_lines)
        
        print(reply)


if __name__ == "__main__":
    main()
