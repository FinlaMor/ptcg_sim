import json
import glob

# Grab the first few files from your scraped/problem folder
problem_files = glob.glob("games\\scrapped\\*.json")[:3]

for file_path in problem_files:
    print(f"\n=== DIAGNOSING: {file_path} ===")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            print(f"Root Data Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Root Level Keys: {list(data.keys())}")
                # Print sample depth of keys to find where the players/decks hide
                for k in list(data.keys())[:3]:
                    if isinstance(data[k], dict):
                        print(f"  -> Keys inside '{k}': {list(data[k].keys())}")
                    elif isinstance(data[k], list) and len(data[k]) > 0:
                        print(f"  -> '{k}' is a list of length {len(data[k])}. First item type: {type(data[k][0])}")
                        if isinstance(data[k][0], dict):
                            print(f"     -> Keys inside first item: {list(data[k][0].keys())}")
            elif isinstance(data, list):
                print(f"Root is a LIST of length {len(data)}.")
                if len(data) > 0 and isinstance(data[0], dict):
                    print(f"  -> Keys inside first list item: {list(data[0].keys())}")
        except Exception as e:
            print(f"Failed to read file: {e}")
