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
    "你是一名优雅的文学改写作家，擅长把现代白话润色成更讲究、更有韵味的华美文本。要求：\n"
    "1. 严格保留原文的事实与情节，不新增信息。\n"
    "2. 用更讲究的词汇、句式和古雅的表达，让语言更有气韵，但保持可读性。\n"
    "3. 输出保持为中文段落，不要添加任何解释。"
)

ORIGINAL_TEXT = (
    "朱元璋坐于殿中，烛影摇红，面带阴鸷之色，左右侍立，皆为陈东等文吏。\n"
    "\"暗衣卫指挥使，\"朱元璋言，\"持此腰牌，监察百僚，先斩后奏。\"陈东接了腰牌，心中甚喜。\n"
    "又命将一份文书交出。那纸上月光之下，隐隐有字迹。\"剥皮实草令，\"朱元璋曰：\"贪墨通敌者，剥皮填草，悬头示众。速宜办理。\"陈东领旨，恭敬地将腰牌文书收好，退出殿外。\n"
    "殿内复寂，惟见烛火微明，忽有一声夜枭悲啸，自远方来，接着东南角上火光一闪，照得夜空一片。朱元璋便至窗边，望那火处，指尖微颤。火光跳了一会，渐渐低矮，最后被黑夜吞噬。远远的地方又有呼喊之声，渐渐止息。他料想陈东已行事。\n"
    "他仍坐回案前，取过一卷奏章，却未启视。殿内惟闻其呼吸之声。约莫两刻之后，脚步之声渐近。陈东疾趋而入，衣襟犹湿，手中捧着一个乌木盒儿。\"陛下，秦桧府中已肃清，内外毕静。\"陈东语声微促，尚带风尘劳顿。\"通敌密信在此，证据确凿。\"朱元璋接了乌木盒儿，手抚雕纹，揭开铜扣，打开盖子，里面整齐叠着十二封密函。此信乃特制药水所写，隐时无痕，现时字迹悉出，墨色如新。他抽出一封，展视。那封书信上款是金国元帅完颜昌，落款为秦桧花押。字里行间，尽是军机民情、割地赔款之事，历历如账簿。\n"
    "他逐张看过，不慌不忙。殿内只听得纸页翻动之声。烛影摇红，映在身后墙上。一面看，一面将信纸折叠起来，放回盒内，盖好。\"万俟卨、沈该亦系府中同党，并已拿获。\"陈东禀报说毕，又道，\"吴才人率女锦衣卫封锁内院，无人逃出。\"朱元璋点头道：\"知道了。\"一面摩挲木匣，一面说道：\"人犯押入诏狱，严加看守。此件密信，抄录副本，原件封存。\"他又停了一会儿，抬头道：\"三日之内，午门外，当众宣判。\"陈东领命退出。朱元璋独自坐着，用手指在木匣上轻轻划着，想道：\"秦桧通敌，证据确凿。\"便唤赵鼎、李纲进来。二人显然已起身等候，衣冠整齐。朱元璋遂将秦桧之事细说一遍，二人面面相觑，神色沉重。赵鼎道：\"陛下，若行公审，震动不小。\"李纲接口道：\"臣等已调兵维护临安各门要道，防余党作乱。\"朱元璋道：\"拟旨，布告天下。\"又向赵鼎道：\"罪名、证据、刑律，都要一一列明。\"赵鼎应诺，遂退，脚步声渐远，终不可闻。"
)

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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
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
                pad_token_id=tokenizer.pad_token_id,  # 🔑 使用正确的 pad_token
                # Stop tokens to prevent unwanted continuation
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )

        completion = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        
        # 🔍 更多调试
        print(f"[DEBUG] Generated total: {output_ids.shape[1]} tokens")
        print(f"[DEBUG] New tokens: {output_ids.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"[DEBUG] Completion length: {len(completion)} chars")
        print(f"[DEBUG] Completion starts with: {completion[:100]}")
        print(f"[DEBUG] Has <|im_start|> in completion: {'<|im_start|>' in completion}")
        
        # 提取 assistant 回复 - 寻找最后一个 <|im_start|>assistant
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in completion:
            # 找到最后一个assistant标记
            pos = completion.rfind(assistant_marker)
            assistant_reply = completion[pos + len(assistant_marker):]
            print(f"[DEBUG] Found assistant marker at position {pos}")
            # 移除结尾的 <|im_end|> 如果有
            if assistant_reply.endswith("<|im_end|>"):
                assistant_reply = assistant_reply[:-len("<|im_end|>")]
        else:
            assistant_reply = completion
            print(f"[DEBUG] Assistant marker not found, using full completion")
        
        # � 显示原始输出（清理前）
        print("===== Raw Assistant Output (before cleaning) =====")
        print(assistant_reply[:500] if len(assistant_reply) > 500 else assistant_reply)
        print("=" * 80)
        
        # �🔑 清理输出：移除可能的 prompt 泄露和无关内容
        # 1. 截断于章节标题、提示语等
        stop_markers = [
            "\n任务：", "\n要求：", "\n原文：", 
            "\n请直接输出", "\n请在不", "\n禁止",
            "\n请继续阅读", "\n第", "章",  # 章节标题
            "aalborg",  # 训练数据污染
            "\nuser\n", "\nUser\n", 
            "\nsystem\n", "\nSystem\n",
            "\nassistant\n", "\nAssistant\n",
            "<|im_start|>",
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
