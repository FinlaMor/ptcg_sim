import os
import csv
import ast

LOG_DIR = "games\\selfplay\\"
OUTPUT_CSV = "upgraded_training_data.csv"

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
CSV_HEADER = SCALAR_COLUMNS + [f"p{player}_{col}" for player in (0, 1) for col in PLAYER_COLUMNS] + ["label"]

def clean_value(col_name, val):
    if isinstance(val, str):
        if val == "ACTIVE":
            return 1
        if col_name == "file":
            val_clean = val.replace(".json", "")
            return int(val_clean) if val_clean.isdigit() else (abs(hash(val)) % 10000)
    if val is None:
        return 0
    return val

def parse_all_logs():
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' does not exist.")
        return

    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.txt')]
    print(f"Scanning {len(log_files)} game log histories...")

    processed_rows = 0

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        for filename in log_files:
            file_path = os.path.join(LOG_DIR, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # --- FIX: Scan through EVERY line to capture mid-game decisions ---
            for i in range(len(lines) - 1):
                current_line = lines[i].strip()
                next_line = lines[i+1].strip()

                # Catch both the initial setups AND subsequent middle game turn states
                is_valid_state = current_line.startswith("START_STATE:") or current_line.startswith("NEXT_STATE:")
                
                if is_valid_state and next_line.startswith("DECISION|"):
                    try:
                        # Strip away the matching header tag dynamically
                        dict_str = current_line.split(":", 1)[1]
                        if dict_str == "None":
                            continue
                            
                        feature_dict = ast.literal_eval(dict_str)

                        # Extract decision target scores cleanly
                        decision_parts = next_line.split("|")
                        score_part = decision_parts[2] # expected_score:X.XXXX
                        target_score = float(score_part.split(":")[1])

                        # Build the matrix row
                        row_data = []
                        for col in CSV_HEADER:
                            if col == "label":
                                row_data.append(target_score)
                            else:
                                raw_val = feature_dict.get(col, 0)
                                row_data.append(clean_value(col, raw_val))

                        writer.writerow(row_data)
                        processed_rows += 1

                    except (ValueError, SyntaxError, IndexError):
                        continue

    print(f"Parsing complete! Expanded your dataset into {processed_rows} total state patterns inside '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    parse_all_logs()
