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
    parser = argparse.ArgumentParser(description="Stage-1 LoRA style injection training")
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
        type=int,
        default=10,
        help="Trainer logging interval.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Checkpoint interval for QA sampling.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use datasets streaming mode for very large corpora.",
    )
    return parser.parse_args()


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


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 4096
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )


def build_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: Path,
    logging_steps: int,
    save_steps: int,
    streaming: bool,
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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
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
        attn_implementation="flash_attention_2",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=None,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        max_seq_length=4096,
        data_collator=collator,
        packing=True,
        peft_config=lora_config,
    )
    return trainer


def main() -> None:
    args = parse_args()
    dataset = load_corpus(args.dataset_path, args.streaming)

    tokenizer = build_tokenizer(args.model_name_or_path)
    model = build_model(args.model_name_or_path)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        streaming=args.streaming,
    )

    trainer.train()
    trainer.save_model()
    trainer.tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
