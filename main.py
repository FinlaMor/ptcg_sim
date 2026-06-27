import lightgbm as lgb
import random
from cg.api import Observation, to_observation_class
import os
from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, search_begin, search_step, search_end, search_release
from time import perf_counter as pf
from datetime import datetime as dt
import sys
import sys
import glob 

# --- METADATA CONFIGURATION LAYER ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
output_path = "games\\selfplay\\"
DECKS_DIR = "decks\\"  # The destination folder from your parser script

def load_all_extracted_decks():
    """Scans the decks folder and returns a list of all available deck file paths."""
    deck_files = glob.glob(os.path.join(DECKS_DIR, "*.txt"))
    if not deck_files:
        # Fallback to your original manual CSV paths if the folder is empty
        print("[NOTICE] No extracted decks found. Defaulting to standard 'deck.csv'.")
        return ["deck.csv"]
    return deck_files

def read_deck_file(file_path) -> list[int]:
    """Reads a text deck file. Returns a list of 60 card IDs."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.read().strip().split("\n")
        deck = [int(line) for line in lines if line.strip().isdigit()]
    # Enforce standard deck compliance padding if files are clipped
    if len(deck) < 60:
        deck += [1] * (60 - len(deck))  # Pad with Basic Energy
    return deck[:60]

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current Working Directory: {os.getcwd()}")
output_path = "games\\selfplay\\"

def read_deck_csv(file_path="deck.csv") -> list[int]:
    """Read deck.csv.
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
        deck = []
        for i in range(60):
            deck.append(int(csv[i]))
    return deck

SCALAR_COLUMNS = [
    "file", "step_index", "entry_index", "status", "select_context", 
    "select_type", "select_option_count", "yourIndex", "turn", 
    "turnActionCount", "firstPlayer", "energyAttached", "supporterPlayed", 
    "stadiumPlayed", "retreated", "result",
]
PLAYER_COLUMNS = [
    "deckCount", "discardCount", "prizeCount", "handCount", "benchCount", 
    "activeExists", "activeHp", "activeMaxHp", "activeEnergyCount", 
    "activeToolCount", "activeAppearThisTurn", "poisoned", "burned", 
    "asleep", "paralyzed", "confused",
]
HEADER = SCALAR_COLUMNS + [f"p{player}_{col}" for player in (0, 1) for col in PLAYER_COLUMNS] + ["label"]

def to_int(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return 0
    return int(value)

def get_val(obj, key, default=None):
    """Safely extracts a value from an object attribute or a dictionary key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def extract_active_features(active_list):
    if not active_list or active_list is None:
        return {
            "activeExists": 0, "activeHp": 0, "activeMaxHp": 0,
            "activeEnergyCount": 0, "activeToolCount": 0, "activeAppearThisTurn": 0,
        }
    active = active_list[0] if isinstance(active_list, list) else active_list
    energies = get_val(active, "energies", [])
    tools = get_val(active, "tools", [])
    return {
        "activeExists": 1,
        "activeHp": to_int(get_val(active, "hp", 0)),
        "activeMaxHp": to_int(get_val(active, "maxHp", 0)),
        "activeEnergyCount": len(energies) if isinstance(energies, list) else 0,
        "activeToolCount": len(tools) if isinstance(tools, list) else 0,
        "activeAppearThisTurn": to_int(get_val(active, "appearThisTurn", False)),
    }

def player_features(player):
    discard = get_val(player, "discard", [])
    prize = get_val(player, "prize", [])
    bench = get_val(player, "bench", [])
    active = get_val(player, "active", [])
    result = {
        "deckCount": to_int(get_val(player, "deckCount", 0)),
        "discardCount": len(discard) if isinstance(discard, list) else 0,
        "prizeCount": len(prize) if isinstance(prize, list) else 0,
        "handCount": to_int(get_val(player, "handCount", 0)),
        "benchCount": len(bench) if isinstance(bench, list) else 0,
        "poisoned": to_int(get_val(player, "poisoned", False)),
        "burned": to_int(get_val(player, "burned", False)),
        "asleep": to_int(get_val(player, "asleep", False)),
        "paralyzed": to_int(get_val(player, "paralyzed", False)),
        "confused": to_int(get_val(player, "confused", False)),
    }
    result.update(extract_active_features(active))
    return result

def extract_features(entry, step_index=1):
    if entry is None:
        return None

    obs = get_val(entry, "observation")
    if obs is not None:
        current = get_val(obs, "current")
        select_dict = get_val(obs, "select", {})
    else:
        current = entry if get_val(entry, "yourIndex") is not None else get_val(entry, "current")
        select_dict = {}

    if current is None:
        return None

    your_index = get_val(current, "yourIndex")
    if your_index is None:
        return None

    file_raw = str(getattr(start_data, "battlePtr", "0"))
    if not file_raw.endswith(".json") and file_raw != "0":
        file_raw = f"{file_raw}.json"

    row = {
        "file": file_raw,
        "step_index": to_int(step_index),
        "entry_index": to_int(your_index),
        "status": str(get_val(entry, "status", "ACTIVE")),
        "select_context": get_val(select_dict, "context"),
        "select_type": get_val(select_dict, "type"),
        "select_option_count": len(get_val(select_dict, "option", [])),
        "yourIndex": to_int(your_index),
        "turn": to_int(get_val(current, "turn", 0)),
        "turnActionCount": to_int(get_val(current, "turnActionCount", 0)),
        "firstPlayer": to_int(get_val(current, "firstPlayer", -1)),
        "energyAttached": to_int(get_val(current, "energyAttached", False)),
        "supporterPlayed": to_int(get_val(current, "supporterPlayed", False)),
        "stadiumPlayed": to_int(get_val(current, "stadiumPlayed", False)),
        "retreated": to_int(get_val(current, "retreated", False)),
        "result": to_int(get_val(current, "result", -1)),
    }

    players = get_val(current, "players", [])
    if not players or len(players) < 2:
        return None

    for player_index in (0, 1):
        player = players[player_index]
        prefix = f"p{player_index}_"
        player_feats = player_features(player)
        for key, value in player_feats.items():
            row[prefix + key] = value

    return row

def agent_blueprint(obs_dict, search_depth=1):
    """
    Upgraded Agent: Safely evaluates the immediate future of every legal move.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    legal_moves = obs.select.option
    best_move = 0 if len(legal_moves) > 0 else 0 
    highest_score = -float('inf')
    
    # --- FIX: Match your 48-feature training configuration ---
    # Only exclude 'label' since it's the target. Keep 'file' and 'step_index'!
    feature_columns = [col for col in HEADER if col != "label"]

    for move in range(len(legal_moves)):
        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active
        
        search = search_begin(
            obs, 
            your_deck=random.sample(deck_0, state.players[your_index].deckCount),
            your_prize=random.sample(deck_0, len(state.players[your_index].prize)),
            opponent_deck=[1072] * state.players[1 - your_index].deckCount,  # Fix: Added [1072]
            opponent_prize=[1] * len(state.players[1 - your_index].prize),   # Fix: Added [1]
            opponent_hand=[1] * state.players[1 - your_index].handCount,     # Fix: Added [1]
            opponent_active=[1072] if len(active) > 0 and active[0] is None else [] # Safe list wrap
        )
        
        try:
            hypothetical_state = search_step(search.searchId, [move])
            board_features = extract_features(hypothetical_state, step_index=1)
            
            ordered_features = []
            for col in feature_columns:
                val = board_features.get(col, 0) if board_features else 0
                if isinstance(val, str):
                    if val == "ACTIVE":
                        val = 1
                    else:
                        val_clean = val.replace(".json", "")
                        val = int(val_clean) if val_clean.isdigit() else (abs(hash(val)) % 10000)
                if val is None:
                    val = 0
                ordered_features.append(val)
                
            if bst is not None:
                score = float(bst.predict([ordered_features])[0])
            else:
                score = 0.5  # Neutral fallback for Cycle 1 baseline exploration
            
        except Exception as e:
            score = 0.0
        finally:
            search_end()
            search_release(search.searchId)
        
        if score > highest_score:
            highest_score = score
            best_move = move
            
    # print(f"Search Best move: {best_move} with expected score: {highest_score:.4f}")
    return best_move, float(highest_score if highest_score != -float('inf') else 0.0)

def stop_watch(st, message):
    time = pf() - st
    print(f"{message} took {time:.4f} seconds")
    return pf()

# Core System Initializer
st = pf()
bst = None
if os.path.exists('rotom_computer.txt'):
    try:
        with open('rotom_computer.txt', 'r', encoding='utf-8') as f:
            model_str = f.read()
        bst = lgb.Booster(model_str=model_str)
        print("[SUCCESS] Loaded existing rotom_computer.txt brain file.")
    except Exception:
        print("[WARNING] Failed to parse rotom_computer.txt. Starting from raw baseline.")
        bst = None
else:
    print("[INITIALIZATION] No rotom_computer.txt found. Agent will play from raw baseline for Cycle 1.")
deck_0_path = "deck.csv"
deck_1_path = "deck.csv"
st = stop_watch(st, "Model loaded")

deck_0 = read_deck_csv(deck_0_path)
deck_1 = read_deck_csv(deck_1_path)
st = stop_watch(st, "Decks loaded")

# Core Match Execution Loop
# --- CONTINUOUS LOOP MODIFICATION ---
# Check if an argument was passed from the batch file; default to infinity if not
if len(sys.argv) > 1:
    max_games = int(sys.argv[1])
    print(f"Looping Pipeline Active: Simulating {max_games} matches...")
else:
    max_games = float('inf')

games_played = 0

# --- INITIALIZE PIPELINE SYSTEM ---
if len(sys.argv) > 1:
    max_games = int(sys.argv[1])
    print(f"Looping Pipeline Active: Simulating {max_games} matches...")
else:
    max_games = float('inf')

games_played = 0

# Scan and cache all available unique decks once at boot time
all_available_decks = load_all_extracted_decks()
print(f"Loaded {len(all_available_decks)} unique archetypes into simulation meta pool.")

while games_played < max_games:
    # 1. Randomly sample two deck paths from your extracted pool
    # Using random.choices handles the case where you only have 1 file smoothly
    deck_0_path = random.choice(all_available_decks)
    deck_1_path = random.choice(all_available_decks)
    
    # 2. Parse the card IDs for this specific game seed
    deck_0 = read_deck_file(deck_0_path)
    deck_1 = read_deck_file(deck_1_path)
    
    # 3. Boot the environment with the dynamically selected lists
    obs, start_data = battle_start(deck_0, deck_1)
    if obs is None:
        print("Battle start failed or simulations complete.")
        break
        
    os.makedirs(output_path, exist_ok=True)
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S_%f')
    file_name = f"{start_data.battlePtr}_{timestamp}.txt"
    file_path = os.path.join(output_path, file_name)
    
    # Print high-level matchup visibility so you can see who is fighting
    p0_name = os.path.basename(deck_0_path).replace("_deck.txt", "")
    p1_name = os.path.basename(deck_1_path).replace("_deck.txt", "")
    # print(f" Match {games_played + 1}/{max_games}: [{p0_name}] vs [{p1_name}]")
    
    with open(file_path, "a") as f:
        f.write(f"START_STATE:{extract_features(obs, step_index=1)}\n")
        
        while True:
            if obs is None or get_val(obs, "select") is None:
                break
                
            move, tree_score = agent_blueprint(obs, search_depth=1)
            f.write(f"DECISION|move:{move}|expected_score:{tree_score:.4f}\n")
            
            try:
                obs = battle_select([move])
                if obs is None:
                    break
                f.write(f"NEXT_STATE:{extract_features(obs, step_index=1)}\n")
            except IndexError:
                break
            
        battle_finish()
    
    games_played += 1

print(f"Completed simulation block of {games_played} games cleanly.")
