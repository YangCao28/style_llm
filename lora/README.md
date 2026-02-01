# LoRA Stage-1 Style Injection

This folder contains the Stage-1 (unsupervised style injection) training stack for the 45k × 1024-token corpus.

## Data expectations
- Input file: `data/dataset/combined_dataset.jsonl` with packed 1024-token segments already cleaned, simplified, and free of metadata.
- Each JSON record must contain at least a `text` field. The script applies a final regex sweep to drop any residual `作者：` lines, chapter headers, or `http://` noise before training.

## Training command (single 80GB A100)
```bash
conda run -p ./.conda python -m lora.stage1_style_injection \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_path data/dataset/combined_dataset.jsonl \
  --output_dir ./stage1_style_injection
```
Or load every hyperparameter from a JSON file:
```bash
conda run -p ./.conda python -m lora.stage1_style_injection \
  --config lora/stage1_style_config.example.json
```
### Quick qualitative test
After a training run finishes, load the adapter and run a short generation to inspect style:
```bash
conda run -p ./.conda python -m lora.test_style_injection \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --lora_path stage1_style_injection \
  --prompt "今日天气阴沉，"
```
Adjust `--prompt`, `--temperature`, or `--max_new_tokens` to survey different behaviors.

Key hyperparameters:
- LoRA: r=128, alpha=256, dropout=0.05, target modules = `q/k/v/o/gate/up/down`.
- Context length: 4096 with packing enabled (4×1024 segments per sample).
- Global batch: 128 tokens (8 per-device × 16 gradient accumulation) using BF16.
- LR = 5e-5, cosine decay, warmup ratio 0.03, single epoch.
- FlashAttention-2 + gradient checkpointing keep memory within 80GB.

## Monitoring checklist
1. **Loss curve** – expect a smooth drop from ≈2.5 to ≤1.8 by epoch end. Plateauing above 2.0 usually means formatting noise still exists in the corpus.
2. **Checkpoint sampling** – checkpoints saved every 500 steps. Run a quick greedy decode with a neutral prompt (e.g., “今日天气阴沉，”); stylistic markers and rare lexicon from the corpus should appear rapidly.
3. **Drift control** – this phase intentionally overwrites polite/conversational priors. Stage-2 SFT will restore instruction-following, so no need to mix in dialogue data here.

## Extending
- Use `--streaming` if you host the JSONL on an object store and want to avoid local disk IO bottlenecks.
- Adjust `per_device_train_batch_size` (up to 12–16) only if your monitoring shows under-utilized VRAM.
- For multi-GPU runs, launch via Accelerate or DeepSpeed; no code changes are needed because the trainer inherits HF TrainingArguments semantics.
- Copy `stage1_style_config.example.json` and edit values to create reproducible launch profiles. Any CLI option overrides the JSON entries, so you can keep multiple presets (e.g., 8B vs 8x7B) without duplicating scripts.
