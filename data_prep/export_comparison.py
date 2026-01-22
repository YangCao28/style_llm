
import json
import argparse
from pathlib import Path

def export_pairs_to_json(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    pairs = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    pairs.append({
                        "id": record.get("id"),
                        "input": record.get("wuxia_text"),
                        "output": record.get("vernacular_text")
                    })
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Exporting {len(pairs)} pairs to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export JSONL pairs to a standard JSON array for comparison.')
    parser.add_argument('--input', default='data_prep/wuxia_vernacular_pairs.jsonl', help='Input JSONL file')
    parser.add_argument('--output', default='data_prep/comparison_dataset.json', help='Output JSON file')
    
    args = parser.parse_args()
    export_pairs_to_json(args.input, args.output)
