import os
import sys
from pathlib import Path

# Add current directory to path so we can import translate_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from translate_agent import LLMAgent
except ImportError:
    # If running from root, maybe need to adjust
    sys.path.append(os.path.join(os.getcwd(), 'data_prep'))
    from translate_agent import LLMAgent

def main():
    source_dir = Path(r"C:\Users\caoya\source\repos\nexus-ng\novels\daming\output\rewriter\novel_rewritten")
    target_dir = source_dir / "improve"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate config relative to this script
    config_path = Path(__file__).parent / "llm_config.json"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    print(f"Using config: {config_path}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")

    agent = LLMAgent(str(config_path))
    
    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        return

    files = list(source_dir.glob("*.txt"))
    print(f"Found {len(files)} txt files.")

    for file_path in files:
        output_path = target_dir / file_path.name
        
        print(f"Processing {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Skipping empty file: {file_path.name}")
                continue

            rewritten_text = agent.generate_vernacular(content)
            
            if rewritten_text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(rewritten_text)
                print(f"Saved to {output_path.name}")
            else:
                print(f"Failed to generate text for {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    main()
