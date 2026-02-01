"""LoRA stage-1 (style injection) training script for 45k x 1024-token corpus.

Key capabilities:
- Streams JSONL data with a single "text" field and performs final cleanup.
- Builds a prompt/response template so only the prose body contributes to loss.
- Configures TRL's SFTTrainer with FlashAttention-2, packing, and LoRA (r=128, alpha=256).
- Targets all linear submodules (q/k/v/o/gate/up/down) to force a strong style bias.
- Optimized for a single 80GB A100: max_seq_length=4096, global batch 128.

Usage (single GPU):
    conda run -p ./.conda python -m lora.stage1_style_injection \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset_path data/dataset/combined_dataset.jsonl

Monitoring:
- Expect loss to fall from ~2.5 to <=1.8 by the end of the single epoch.
- Checkpoints every 500 steps support short-form sampling QA during training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
RESPONSE_TEMPLATE = "### 正文\n"
PROMPT_TEMPLATE = "### 文风语料\n"
MERGED_TOKEN_STATEMENT = "【packed_style_sample】\n"

# Regexes for stripping residual metadata such as 作者:xxx, http links, chapter labels.
META_PATTERNS = [
    re.compile(r"作者[:：].*?$", re.MULTILINE),
    re.compile(r"^\s*第[一二三四五六七八九十百千0-9]+[章回节].*?$", re.MULTILINE),
    re.compile(r"https?://\S+", re.IGNORECASE),
]

WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Stage-1 LoRA style injection training", parents=[config_parser])
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/dataset/combined_dataset.jsonl"),
        help="JSONL file with a single 'text' column (4096-token packed entries).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./stage1_style_injection"),
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--logging_steps",
        parser.add_argument(
            "--attn_impl",
            type=str,
            default="flash_attention_2",
            help="Attention implementation passed to HF model (default: flash_attention_2)",
        )

        if config_args.config:
            if not config_args.config.exists():
                raise FileNotFoundError(config_args.config)
            with config_args.config.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
            normalized_defaults = {}
            for key, raw_value in config_data.items():
                matched_action = next((action for action in parser._actions if action.dest == key), None)
                if matched_action is None:
                    raise ValueError(f"Unknown config key '{key}' in {config_args.config}")
                converter = matched_action.type
                if converter and raw_value is not None and not isinstance(raw_value, converter):
                    normalized_defaults[key] = converter(raw_value)
                else:
                    normalized_defaults[key] = raw_value
            parser.set_defaults(**normalized_defaults)

        args = parser.parse_args(remaining_argv)
        args.config = config_args.config or args.config
        return args
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        help="Attention implementation passed to HF model (default: flash_attention_2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Any CLI flag overrides entries inside the JSON.",
    )
    args = parser.parse_args()
    if args.config:
        args = apply_json_config(args, args.config)
    return args


def apply_json_config(args: argparse.Namespace, config_path: Path) -> argparse.Namespace:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in config.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown config key '{key}' in {config_path}")
        current = getattr(args, key)
        # CLI arguments take precedence; only override defaults untouched by user.
        parser_default = parser_default_values().get(key)
        if args.config and key == "config":
            continue
        if current == parser_default or key == "config":
            setattr(args, key, value)
    return args


def parser_default_values() -> Dict[str, object]:
    return {
        "model_name_or_path": DEFAULT_MODEL,
        "dataset_path": Path("data/dataset/combined_dataset.jsonl"),
        "output_dir": Path("./stage1_style_injection"),
        "logging_steps": 10,
        "save_steps": 500,
        "streaming": False,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.03,
        "num_train_epochs": 1.0,
        "max_seq_length": 4096,
        "attn_impl": "flash_attention_2",
        "config": None,
    }


def load_corpus(dataset_path: Path, streaming: bool) -> Dataset:
    if not dataset_path.exists() and not streaming:
        raise FileNotFoundError(dataset_path)

    data_files = str(dataset_path)
    ds = load_dataset("json", data_files=data_files, split="train", streaming=streaming)
    return ds


def normalize_text(record: Dict[str, str]) -> str:
    text = record.get("text", "")
    for pattern in META_PATTERNS:
        text = pattern.sub("", text)
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    if not text:
        return ""
    # Ensure consistent paragraph spacing for packed training.
    paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n\n".join(paragraphs)


def formatting_func(record: Dict[str, str]) -> str:
    body = normalize_text(record)
    if not body:
        return ""
    return f"{PROMPT_TEMPLATE}{MERGED_TOKEN_STATEMENT}{RESPONSE_TEMPLATE}{body}\n"


def build_tokenizer(model_name: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name: str, attn_impl: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
    )


def build_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: Path,
    logging_steps: int,
    save_steps: int,
    streaming: bool,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    num_train_epochs: float,
    max_seq_length: int,
    attn_impl: str,
):
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=5,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to=("tensorboard",),
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        attn_implementation=attn_impl,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=None,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        data_collator=collator,
        packing=True,
        peft_config=lora_config,
    )
    return trainer


def main() -> None:
    args = parse_args()
    dataset = load_corpus(args.dataset_path, args.streaming)

    tokenizer = build_tokenizer(args.model_name_or_path, args.max_seq_length)
    model = build_model(args.model_name_or_path, args.attn_impl)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        streaming=args.streaming,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        attn_impl=args.attn_impl,
    )

    trainer.train()
    trainer.save_model()
    trainer.tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
