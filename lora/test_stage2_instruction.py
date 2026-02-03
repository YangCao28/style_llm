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
    "你是一个专业的文学改写工具，仅执行文风转换任务。"
    "你的唯一职责是：对给定文本进行文风调整，保持内容完全一致。"
    "严格禁止：(1)添加任何原文不存在的信息、情节、人物、对话；(2)删减或略去任何原文信息；"
    "(3)在改写结果后继续生成任何内容；(4)生成任何解释、评论、提示或元信息。"
    "输出格式：直接给出改写后的文本，不包含任何其他内容。改写完成后立即停止生成。"
    "⚠️ 禁止输出改写内容以外的任何字，违者报错。输出完改写结果后必须立即终止，不得继续任何形式的文本生成。"
)

ORIGINAL_TEXT = (
    "岳飞年方二十六，旧甲犹存，累有战功，不露得意之色。韩世忠四十余岁，老于军旅，梁红玉二十七岁，眉目间有水战将领的精明之气。黄纵监制新式火药，炮内实铁弹。炮口对准三里外一道夯土墙。\"放！\"\n"
    "引信咝咝烧尽，插入炮底。一声轰雷，白烟冲天，铁弹破空，直飞土墙。\n"
    "土墙倒塌，烟尘弥漫。岳飞首先来视，他蹲下身去，指量坑洞之深广，点头称许：\"足矣。倘使齐发，则足以破重甲。\"韩世忠测射程，梁红玉察轨道。三人在旁细商，又试火铳。火铳射程百步，铅弹能穿铁甲。\n"
    "朱元璋望见那崩塌的土墙，烟尘在风中渐渐散去。\"组神机营，\"他命道，\"黄纵督办火器，岳飞、韩世忠、梁红玉教练。秋收以前，我要看见三千人能列队施火器。\"众人领旨而去。\n"
    "出得府库，忽见深处有几口樟木箱，锁扣上尚有新痕，在昏暗之中，尤见分明。\"何物？\"他问。\n"
    "黄纵便随他瞧了，因说道：\"连日清查旧械，仓忙之至，撬锁之际，未及细查。\" 朱元璋也不问，转身出去了。外面天色已晚，临安城里的灯笼都已挂上。秦桧已伏诛。江南田亩正加紧丈量。军器局的火器图样，已交工部。接着就是整顿临安禁军，户部须在秋成以前，编定新税册。"
)

DEFAULT_USER = (
    "任务：将以下文本改写为紧张、悬疑的文风。\n"
    "要求：(1)严格保持所有信息点不变；(2)只改变表达方式和气氛；(3)禁止添加任何新内容。\n"
    "原文：" + ORIGINAL_TEXT + "\n"
    "请直接输出改写结果，不要有任何其他内容："
)


PRESET_CASES = {
    "ni_kuang": {
        "system": DEFAULT_SYSTEM,
        "user": DEFAULT_USER,
        "description": "悬疑风格改写测试（严格禁止增删）",
    },
    "zhang_henshui": {
        "system": (
            "你是一个专业的文学改写工具，仅执行文风转换任务。"
            "你的唯一职责是：将文本改写为市井生活、细腻情感风格，保持内容完全一致。"
            "严格禁止：(1)新增任何情节、人物、对话；(2)删减原文信息；(3)改写后继续生成内容。"
            "输出格式：直接给出改写后的文本，改写完成后立即停止。"
            "⚠️ 禁止输出改写内容以外的任何字，违者报错。"
        ),
        "user": (
            "任务：将以下文本改写为市井情调、细腻情感的文风。\n"
            "要求：(1)严格保持所有信息点不变；(2)只改变表达方式；(3)禁止添加任何新内容。\n"
            "原文：" + ORIGINAL_TEXT + "\n"
            "请直接输出改写结果，不要有任何其他内容："
        ),
        "description": "市井情感风格改写测试（严格禁止增删）",
    },
    "plain_to_style": {
        "system": DEFAULT_SYSTEM,
        "user": (
            "任务：将以下朴素叙述改写为悬疑气氛的文风。\n"
            "要求：(1)严格保持所有信息点不变；(2)只改变气氛和语气；(3)禁止添加任何新内容。\n"
            "原文：" + ORIGINAL_TEXT + "\n"
            "请直接输出改写结果，不要有任何其他内容："
        ),
        "description": "悬疑气氛加强测试（严格禁止增删）",
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
            
            # 2. 尝试从 config.json 读取 _name_or_path（可能是本地路径）
            if not base_model_path:
                config_path = model_path / "config.json"
                if config_path.exists():
                    import json
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        base_model_path = config.get("_name_or_path")
                        if base_model_path:
                            print(f"  从 config.json 读取: {base_model_path}")
                            # 如果是相对路径，转换为绝对路径
                            if base_model_path and not base_model_path.startswith("/") and "/" not in base_model_path[:10]:
                                base_model_path = str((model_path.parent / base_model_path).resolve())
            
            # 3. 尝试从父目录或祖父目录找 Stage1 模型（本地路径）
            if not base_model_path:
                # stage2_instruction_tuning_corrected/checkpoint-16 -> stage1_style_injection
                parent_dir = model_path.parent.parent
                possible_stage1_paths = [
                    parent_dir / "stage1_style_injection",
                    parent_dir.parent / "stage1_style_injection",  # 再往上一层
                ]
                for stage1_path in possible_stage1_paths:
                    if stage1_path.exists():
                        # 直接使用 Stage1 路径（包含 tokenizer）
                        print(f"  找到 Stage1 模型: {stage1_path}")
                        base_model_path = str(stage1_path.resolve())
                        break
            
            # 4. 尝试查找本地 Qwen 模型目录
            if not base_model_path:
                # 常见的本地路径
                possible_local_paths = [
                    Path("/workspace/models/Qwen2.5-8B-Base"),
                    Path("/workspace/models/Qwen2.5-7B-Base"),
                    Path("./models/Qwen2.5-8B-Base"),
                    Path("../models/Qwen2.5-8B-Base"),
                ]
                print("  尝试本地模型路径...")
                for local_path in possible_local_paths:
                    if local_path.exists() and (local_path / "tokenizer_config.json").exists():
                        print(f"    找到: {local_path}")
                        base_model_path = str(local_path.resolve())
                        break
            
            if not base_model_path:
                raise ValueError(
                    "无法确定基础模型名称或路径。\n"
                    "请使用 --base_model_name 参数指定本地路径或 HF 模型名称，\n"
                    "例如: --base_model_name /workspace/models/Qwen2.5-8B-Base\n"
                    "或者: --base_model_name stage1_style_injection"
                )
            
            print(f"  Loading tokenizer from: {base_model_path}")
            # 尝试作为本地路径，如果失败则作为 HF 模型名称
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
                print("  ✓ 从本地加载成功")
            except Exception:
                print("  ⚠ 本地加载失败，尝试从 HuggingFace...")
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
        # print("----- System -----")
        # print(system)
        # print("----- User -----")
        # print(user)
        # print("----- Raw Prompt (truncated) -----")
        # print(prompt[:400] + ("..." if len(prompt) > 400 else ""))
        print("=" * 80)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 🔍 调试信息
        print(f"\n[DEBUG] Prompt length: {len(prompt)} chars, {inputs['input_ids'].shape[1]} tokens")
        print(f"[DEBUG] Prompt ends with: ...{prompt[-100:]}")

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
        
        # 🔍 更多调试
        print(f"[DEBUG] Generated total: {output_ids.shape[1]} tokens")
        print(f"[DEBUG] New tokens: {output_ids.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"[DEBUG] Completion length: {len(completion)} chars")
        print(f"[DEBUG] Completion starts with: {completion[:100]}")
        
        # 提取 assistant 回复
        if prompt in completion:
            assistant_reply = completion[len(prompt):]
            print(f"[DEBUG] Extracted by removing prompt, length: {len(assistant_reply)}")
        else:
            assistant_reply = completion
            print(f"[DEBUG] Prompt not found in completion, using full completion")
        
        # � 显示原始输出（清理前）
        print("===== Raw Assistant Output (before cleaning) =====")
        print(assistant_reply[:500] if len(assistant_reply) > 500 else assistant_reply)
        print("=" * 80)
        
        # �🔑 清理输出：移除可能的 prompt 泄露和无关内容
        # 1. 在第一个出现的 "任务："、"要求："、"原文："、"请直接输出" 等处截断
        stop_markers = [
            "\n任务：", "\n要求：", "\n原文：", 
            "\n请直接输出", "\n请在不", "\n禁止",
            "\nuser\n", "\nUser\n", 
            "\nsystem\n", "\nSystem\n",
            "\nassistant\n", "\nAssistant\n",
            "<|im_start|>", "<|im_end|>",
        ]
        
        for marker in stop_markers:
            if marker in assistant_reply:
                pos = assistant_reply.find(marker)
                assistant_reply = assistant_reply[:pos]
                break
        
        # 2. 去除结尾的不完整句子（如果以标点结束则保留）
        assistant_reply = assistant_reply.strip()
        
        print("===== Assistant Reply =====")
        print(assistant_reply)


if __name__ == "__main__":
    main()
