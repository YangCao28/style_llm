import json
import time
import requests
import argparse
from pathlib import Path

class LLMAgent:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        
        # Support user's nested structure
        self.gen_config = self.config.get('generation', self.config)
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gen_config['api_key']}"
        }
        
    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_vernacular(self, wuxia_text):
        # Construct endpoint
        base_url = self.gen_config.get('base_url', self.gen_config.get('api_endpoint'))
        if not base_url.endswith('/chat/completions'):
            # Append path if the user gave a base domain like https://api.deepseek.com
             url = f"{base_url.rstrip('/')}/chat/completions"
        else:
             url = base_url

        payload = {
            "model": self.gen_config['model'],
            "messages": [
                {"role": "system", "content": self.config['system_prompt']},
                {"role": "user", "content": wuxia_text}
            ],
            "temperature": self.gen_config.get('temperature', 0.7),
            "max_tokens": self.gen_config.get('max_tokens', 1024),
            "stream": False
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.gen_config.get('timeout', 30)
                )
                response.raise_for_status()
                result = response.json()
                # Assuming OpenAI-compatible format
                content = result['choices'][0]['message']['content']
                return content
            except Exception as e:
                print(f"Request failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)
        
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Max records to process this run')
    args = parser.parse_args()

    # Paths
    config_path = 'llm_config.json'
    input_file = Path('data_prep/wuxia_chunks_cleaned.jsonl')
    output_file = Path('data_prep/wuxia_vernacular_pairs.jsonl')
    
    # Initialize Agent
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return

    agent = LLMAgent(config_path)
    
    # Resume logic: Check how many lines processed
    processed_ids = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    pass
    print(f"Resuming... {len(processed_ids)} already processed.")

    # Processing loop
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        session_processed_count = 0
        
        for line in infile:
            if args.limit and session_processed_count >= args.limit:
                print(f"Reached session limit of {args.limit} records. Stopping.")
                break
                
            if not line.strip():
                continue
                
            record = json.loads(line)
            chunk_id = record['id']
            
            if chunk_id in processed_ids:
                continue
            
            wuxia_text = record['text']
            
            print(f"Processing {chunk_id}...")
            vernacular_text = agent.generate_vernacular(wuxia_text)
            
            if vernacular_text:
                new_record = {
                    "id": chunk_id,
                    "author": record['author'],
                    "source_file": record['source_file'],
                    "wuxia_text": wuxia_text,
                    "vernacular_text": vernacular_text
                }
                
                outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                # Optional: Force flush to ensure data is saved immediately
                outfile.flush() 
                session_processed_count += 1
            else:
                print(f"Skipping {chunk_id} due to API failure.")

if __name__ == "__main__":
    main()
