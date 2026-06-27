import os
import json
import hashlib

# --- CONFIGURATION ---
SOURCE_DIR = "games\\scrapped\\"  # Folder containing .json match files
OUTPUT_DIR = "decks\\"  # Folder where your unique .txt decks will be saved

def get_deck_hash(card_ids):
    """Generates a unique signature hash for a sorted deck list to find duplicates."""
    # Sorting ensures that the same 60 cards in a different order hash identically
    sorted_cards = sorted(card_ids)
    deck_string = ",".join(map(str, sorted_cards))
    return hashlib.md5(deck_string.encode('utf-8')).hexdigest()

def find_players_block_recursively(data):
    """
    Recursively crawls any data structure (dict or list) to automatically
    locate the initial 'players' list setup regardless of its nesting level.
    """
    if isinstance(data, dict):
        # Target a list named 'players' where the first element contains card profiles
        if "players" in data and isinstance(data["players"], list) and len(data["players"]) > 0:
            first_player = data["players"][0]
            if isinstance(first_player, dict) and ("deck" in first_player or "deckCount" in first_player):
                return data["players"]
        
        # Keep searching down all dictionary keys
        for key, value in data.items():
            result = find_players_block_recursively(value)
            if result:
                return result
                
    elif isinstance(data, list):
        # Keep searching down all elements inside an array
        for item in data:
            result = find_players_block_recursively(item)
            if result:
                return result
                
    return None

def extract_decks_from_json(file_path):
    """
    Parses a Kaggle tournament episode JSON and extracts the starting 60-card decks.
    Uses recursive element crawling to handle any unexpected layout variations.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON syntax in: {file_path}")
            return None, None

    # Automatically hunt down the players data cluster across the entire object structure
    players_data = find_players_block_recursively(data)

    if players_data is None:
        print(f"[WARNING] Schema layout unexpected or incomplete in: {file_path}")
        return None, None

    # Extract clean list profiles safely
    decks = {}
    for p_idx in (0, 1):
        try:
            if p_idx < len(players_data):
                player_profile = players_data[p_idx]
                raw_deck_list = player_profile.get("deck", [])
                
                card_ids = [int(card["id"]) for card in raw_deck_list if "id" in card]
                
                if card_ids and len(card_ids) == 60:
                    decks[p_idx] = card_ids
        except (IndexError, KeyError, TypeError):
            continue

        return decks.get(0, None), decks.get(1, None)

    # Clean data profiling block
    decks = {}
    for p_idx in (0, 1):
        try:
            player_profile = players_data[p_idx]
            raw_deck_list = player_profile.get("deck", [])
            
            card_ids = [int(card["id"]) for card in raw_deck_list if "id" in card]
            
            if card_ids and len(card_ids) == 60:
                decks[p_idx] = card_ids
        except (IndexError, KeyError, TypeError):
            continue

    return decks.get(0, None), decks.get(1, None)

def run_deck_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory '{SOURCE_DIR}' missing.")
        return

    json_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]
    print(f"Scanning {len(json_files)} match logs for unique deck configurations...")

    # Set to keep track of deck hashes we have already processed
    seen_deck_hashes = set()
    unique_deck_count = 0
    duplicate_skipped_count = 0

    for filename in json_files:
        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(SOURCE_DIR, filename)
        
        p0_deck, p1_deck = extract_decks_from_json(file_path)
        
        # Process both players' decks through the hash filter
        for player_idx, deck_list in [("player0", p0_deck), ("player1", p1_deck)]:
            if deck_list:
                deck_hash = get_deck_hash(deck_list)
                
                if deck_hash not in seen_deck_hashes:
                    # New unique deck found! Save it.
                    seen_deck_hashes.add(deck_hash)
                    unique_deck_count += 1
                    
                    out_filename = f"{base_name}_{player_idx}_deck.txt"
                    out_path = os.path.join(OUTPUT_DIR, out_filename)
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(map(str, deck_list)))
                else:
                    duplicate_skipped_count += 1

    print(f"\n===================================================")
    print(f"DEDUPLICATION PIPELINE COMPLETE")
    print(f"===================================================")
    print(f"| Unique Decks Saved:  {unique_deck_count}")
    print(f"| Duplicates Filtered: {duplicate_skipped_count}")
    print(f"===================================================")

if __name__ == "__main__":
    run_deck_pipeline()
