import json
from pathlib import Path

def parse_json(json_string):
    """
    Parses a JSON string and returns the corresponding Python object.

    Args:
        json_string (str): A string representation of a JSON object."""
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    
for file in Path("games/scrapped").glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        steps = data.get("steps", [])
        for step in steps:
            print(f"{step}")
            input("Press Enter to continue...")
    print(f"Parsed data from {file}: {data}")
    input("Press Enter to continue...")