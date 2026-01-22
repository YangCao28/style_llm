# Wuxia Novel Dataset Preparation Pipeline

This project contains a set of tools to process a large collection of Wuxia novels (Ni Kuang, Gu Long, etc.) into a clean dataset of Wuxia vs. Vernacular Chinese pairs for LLM fine-tuning.

## Project Structure

```
data-pre/
├── data/                       # Data files (Source & Generated)
│   ├── wuxia_chunks_cleaned.jsonl  # Cleaned chunks
│   └── wuxia_vernacular_pairs.jsonl# Final paired dataset (Output)
├── data_prep/                  # Scripts and Logic
│   ├── convert_files.py            # Step 1: Format and organize raw files
│   ├── prepare_chunks.py           # Step 2: Chunk text into training samples
│   ├── clean_chunks.py             # Step 3: Remove website noise (nav, ads)
│   ├── translate_agent.py          # Step 4: LLM Agent for data synthesis
│   ├── export_comparison.py        # Utility: Export JSONL to JSON for review
│   └── llm_config.json             # Configuration for LLM API
└── requirements.txt            # Python dependencies
```

## Installation

1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow Overview

### 1. Data Standardization (`convert_files.py`)
Reads from the `data/` directory (assumed to contain author folders).
- Converts all text files to **UTF-8**.
- Moves them to a flat directory structure (processed in memory or temporary location).
- Renames files to `{Author}_{Filename}.txt`.
- Merges broken lines (removes hard wraps unless ending with `。`).

```bash
python data_prep/convert_files.py
```

### 2. Chunking (`prepare_chunks.py`)
Reads processed files.
- Splits text into proper sentences.
- Aggregates sentences into chunks of ~800 characters (safe for 1024 token context).
- Outputs to `data/wuxia_chunks_cleaned.jsonl`.

```bash
python prepare_chunks.py
```

### 3. Cleaning (`clean_chunks.py`)
Removes common "web scraping" artifacts like "Next Page", "Download TXT", etc., ensuring clean training data.

```bash
python clean_chunks.py
```

### 4. Synthetic Data Generation (`translate_agent.py`)
Uses an LLM (configured in `llm_config.json`) to rewrite the Wuxia chunks into modern Vernacular Chinese.
- **Input:** `data_prep/wuxia_chunks_cleaned.jsonl`
- **Output:** `data_prep/wuxia_vernacular_pairs.jsonl`
- Supports resume capability (skips already processed IDs).

```bash
python translate_agent.py
```

## Configuration

Modify `llm_config.json` to set your LLM provider details:

```json
{
    "generation": {
        "base_url": "https://api.deepseek.com",
        "api_key": "YOUR_API_KEY",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 8000
    },
    "system_prompt": "..."
}
```
